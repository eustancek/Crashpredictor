# api/app.py - Barebones WSGI application for maximum efficiency
import os
import json
import psutil
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras import backend as K
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashPredictorAPI")

# Configuration
HF_REPO_ID = "eustancek/Google-colab"
MODEL_PATH = "models/latest_model.keras"
PORT = int(os.environ.get("PORT", 8000))

# Global variables
model = None
last_model_load = None
last_warmup = None

def load_model():
    """Download and load the model from Hugging Face with memory monitoring"""
    global model, last_model_load
    
    # Only reload model if it's been more than 1 hour
    from datetime import datetime, timedelta
    now = datetime.now()
    if model is not None and last_model_load and (now - last_model_load) < timedelta(hours=1):
        return model
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    try:
        # Download from Hugging Face
        logger.info("Downloading model from Hugging Face...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_PATH,
            local_dir="."
        )
        
        # Load the model
        logger.info("Loading model...")
        model = tf.keras.models.load_model("models/latest_model.keras")
        last_model_load = datetime.now()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def check_memory():
    """Enhanced memory monitoring with detailed reporting"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    memory_rss = memory_info.rss / 1024 / 1024  # MB
    memory_vms = memory_info.vms / 1024 / 1024  # MB
    
    logger.info(f"Memory usage: RSS={memory_rss:.2f}MB, VMS={memory_vms:.2f}MB ({memory_percent:.2f}%)")
    
    # More detailed memory thresholds
    if memory_percent > 90:
        logger.critical("CRITICAL MEMORY PRESSURE - SERVICE WILL BE UNAVAILABLE")
        return False
    elif memory_percent > 85:
        logger.warning("HIGH MEMORY PRESSURE - SERVICE MAY BECOME UNSTABLE")
        return False
    elif memory_percent > 75:
        logger.info("ELEVATED MEMORY USAGE - MONITORING CLOSELY")
    
    return memory_percent < 85  # Need 15% buffer

def clear_memory():
    """Clear TensorFlow session and Python garbage collection"""
    K.clear_session()
    import gc
    gc.collect()
    logger.debug("Memory cleared after prediction")

def predict(values):
    """Predict the next crash value with memory safety"""
    # Memory check first
    if not check_memory():
        logger.warning("Memory pressure - service temporarily busy")
        return {
            "error": "Service temporarily busy",
            "details": "Please try again in a moment"
        }, 503
    
    try:
        # Load model
        model = load_model()
        if model is None:
            logger.error("Model failed to load")
            return {"error": "Model failed to load"}, 500
            
        # Process input
        input_data = np.array(values, dtype=np.float32).reshape(1, 50, 1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Clear memory after prediction
        clear_memory()
        
        return {"prediction": float(prediction[0][0])}, 200
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}, 500

def warmup():
    """Warmup endpoint to keep the service active"""
    global last_warmup
    
    try:
        # Load the model (this is what keeps the service warm)
        load_model()
        
        # Record last warmup time
        last_warmup = datetime.now()
        logger.info("Service warmed up successfully")
        
        return {
            "status": "warmed up",
            "last_warmup": last_warmup.isoformat(),
            "model_loaded": model is not None
        }, 200
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return {"error": str(e)}, 500

def health_check():
    """Health check endpoint with memory info"""
    process = psutil.Process()
    memory_usage = process.memory_percent()
    memory_rss = process.memory_info().rss / 1024 / 1024  # MB
    return {
        "status": "healthy",
        "memory_usage": f"{memory_usage:.2f}%",
        "memory_rss": f"{memory_rss:.2f}MB",
        "buffer_space": f"{512 - memory_rss:.0f}MB",
        "model_loaded": model is not None,
        "last_model_load": last_model_load.isoformat() if last_model_load else "N/A",
        "last_warmup": last_warmup.isoformat() if last_warmup else "N/A",
        "timestamp": datetime.now().isoformat()
    }

# WSGI application
def application(environ, start_response):
    """WSGI application for minimal memory usage"""
    path = environ['PATH_INFO']
    method = environ['REQUEST_METHOD']
    
    try:
        # Health check endpoint
        if path == '/health' and method == 'GET':
            response = json.dumps(health_check())
            start_response('200 OK', [('Content-Type', 'application/json')])
            return [response.encode()]
        
        # Warmup endpoint (CRITICAL for preventing cold starts)
        elif path == '/warmup' and method == 'GET':
            response_data, status_code = warmup()
            response = json.dumps(response_data)
            start_response(f'{status_code} {status_code_to_message(status_code)}', 
                          [('Content-Type', 'application/json')])
            return [response.encode()]
        
        # Prediction endpoint
        elif path == '/predict' and method == 'POST':
            # Parse input
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size)
                data = json.loads(request_body)
            except Exception as e:
                logger.error(f"Request parsing failed: {str(e)}")
                response = json.dumps({
                    "error": "Invalid request",
                    "details": str(e)
                })
                start_response('400 Bad Request', [('Content-Type', 'application/json')])
                return [response.encode()]
            
            # Validate input
            if 'values' not in data or len(data['values']) != 50:
                logger.warning(f"Invalid input format: expected 50 values, got {len(data.get('values', []))}")
                response = json.dumps({
                    "error": "Invalid input", 
                    "details": "Expected 50 values"
                })
                start_response('400 Bad Request', [('Content-Type', 'application/json')])
                return [response.encode()]
            
            # Make prediction
            response_data, status_code = predict(data['values'])
            response = json.dumps(response_data)
            start_response(f'{status_code} {status_code_to_message(status_code)}', 
                          [('Content-Type', 'application/json')])
            return [response.encode()]
        
        # Unknown endpoint
        else:
            logger.warning(f"Unknown endpoint requested: {path}")
            start_response('404 Not Found', [('Content-Type', 'text/plain')])
            return [b'Not Found']
            
    except Exception as e:
        logger.exception("Unexpected error in WSGI application")
        response = json.dumps({"error": str(e)})
        start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
        return [response.encode()]

def status_code_to_message(code):
    """Convert status code to HTTP message"""
    messages = {
        200: 'OK',
        400: 'Bad Request',
        404: 'Not Found',
        500: 'Internal Server Error',
        503: 'Service Unavailable'
    }
    return messages.get(code, 'Unknown Status')
