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
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashPredictorAPI")

# Must have this import (verify it's working)
try:
    # Try importing from parent directory
    sys.path.append(str(Path(__file__).parent.parent))
    from models.predictor import CrashPredictor
    logger.info("Successfully imported CrashPredictor from models/predictor.py")
except ImportError as e:
    logger.error(f"Failed to import CrashPredictor: {str(e)}")
    raise

# Configuration - CHANGED TO PORT 8080
HF_REPO_ID = "eustancek/Google-colab"
MODEL_PATH = "models/latest_model.keras"
PORT = int(os.environ.get("PORT", 8080))  # CHANGED FROM 8000 TO 8080

# Global variables
predictor = None
last_predictor_load = None
last_warmup = None

def get_predictor():
    """Get or initialize the CrashPredictor instance with memory monitoring"""
    global predictor, last_predictor_load
    
    # Only reload predictor if it's been more than 1 hour
    from datetime import datetime, timedelta
    now = datetime.now()
    if predictor is not None and last_predictor_load and (now - last_predictor_load) < timedelta(hours=1):
        return predictor
    
    try:
        # Initialize predictor
        predictor = CrashPredictor()
        
        # Try to load persistent model
        if predictor.load_persistent_model():
            logger.info("Successfully loaded persistent model")
        else:
            logger.warning("Could not load persistent model, using default")
            
        last_predictor_load = datetime.now()
        return predictor
    except Exception as e:
        logger.error(f"Predictor initialization failed: {str(e)}")
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
        # Load predictor
        predictor = get_predictor()
        if predictor is None:
            logger.error("Predictor failed to initialize")
            return {"error": "Prediction service unavailable"}, 500
            
        # Process input
        if not isinstance(values, list) or len(values) != 50:
            logger.warning(f"Invalid input format: expected 50 values, got {len(values)}")
            return {
                "error": "Invalid input",
                "details": "Expected 50 numerical values"
            }, 400
        
        # Convert to float and handle potential errors
        try:
            input_data = [float(x) for x in values]
        except (TypeError, ValueError) as e:
            logger.warning(f"Input conversion error: {str(e)}")
            return {
                "error": "Invalid input format",
                "details": "All values must be numerical"
            }, 400
        
        # Make prediction
        result = predictor.predict({"multipliers": input_data})
        
        # Clear memory after prediction
        clear_memory()
        
        return result, 200
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}, 500

def warmup():
    """Warmup endpoint to keep the service active"""
    global last_warmup
    
    try:
        # Load the predictor (this is what keeps the service warm)
        get_predictor()
        
        # Record last warmup time
        last_warmup = datetime.now()
        logger.info("Service warmed up successfully")
        
        return {
            "status": "warmed up",
            "last_warmup": last_warmup.isoformat(),
            "predictor_loaded": predictor is not None
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
        "predictor_loaded": predictor is not None,
        "last_predictor_load": last_predictor_load.isoformat() if last_predictor_load else "N/A",
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

# WSGI server entry point (critical for Zeabur)
if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    logger.info(f"Starting server on port {PORT}...")
    server = make_server('0.0.0.0', PORT, application)
    server.serve_forever()
