import os
import numpy as np
import joblib
import time
import psutil
import threading
import glob
import re
import random
import requests
from flask import Flask, request, jsonify
from supabase import create_client
from datetime import datetime, timedelta
import logging
import gc
import tensorflow as tf
from functools import wraps
import queue

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CrashPredictor")

# Initialize Flask app
app = Flask(__name__)

# Supabase configuration
SUPABASE_URL = "https://fawcuwcqfwzvdoalcocx.supabase.co"
SUPABASE_KEY = "DMMWovTqFUm5RAfY"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global predictor instance (initialized on startup)
predictor = None
last_prediction_time = None
MODEL_PREFIX = "model_v"
MODEL_EXTENSION = ".pkl"
MAX_MODELS_TO_KEEP = 1  # Only keep the latest model

# Advanced warm-up configuration
WARMUP_INTERVAL = 240  # 4 minutes (below HF's 5-minute threshold)
WARMUP_JITTER = 60     # Random jitter to avoid predictable patterns
MIN_WARMUP_INTERVAL = 120  # Never go below 2 minutes
MAX_WARMUP_INTERVAL = 300  # Never exceed 5 minutes
WARMUP_TIMEOUT = 15    # Seconds to wait for warm-up to complete
WARMUP_RETRIES = 3     # Number of retry attempts
WARMUP_BACKOFF_FACTOR = 1.5  # Exponential backoff factor
CIRCUIT_BREAKER_THRESHOLD = 5  # Failures before opening circuit
CIRCUIT_BREAKER_RESET_TIME = 300  # 5 minutes to reset circuit

# Circuit breaker state
circuit_open = False
circuit_failure_count = 0
circuit_open_time = None

# Warm-up history for analytics
warmup_history = []
MAX_HISTORY_SIZE = 100

# Warm-up queue for controlled execution
warmup_queue = queue.Queue(maxsize=1)
keep_alive_active = True

class OptimizedPredictor:
    def __init__(self):
        self.model = None
        self.multiplier_range = 0.20
        self.best_accuracy = 75.0
        self.sequence_length = 50
        self.current_version = None
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the latest model version and clean up old models"""
        logger.info("🔄 Searching for available models...")
        
        # Find all model files
        model_files = glob.glob(f"{MODEL_PREFIX}*[0-9]{MODEL_EXTENSION}")
        if not model_files:
            logger.error("❌ No model files found. Using fallback initialization.")
            self._initialize_fallback_model()
            return False
        
        # Extract version numbers and sort
        model_versions = []
        for file in model_files:
            try:
                # Extract version number from filename (e.g., "model_v123.pkl" -> 123)
                version = int(re.search(rf"{MODEL_PREFIX}(\d+){MODEL_EXTENSION}", file).group(1))
                model_versions.append((version, file))
            except (AttributeError, ValueError):
                continue
        
        if not model_versions:
            logger.error("❌ No valid model versions found. Using fallback initialization.")
            self._initialize_fallback_model()
            return False
        
        # Sort by version number (highest first)
        model_versions.sort(key=lambda x: x[0], reverse=True)
        latest_version, latest_file = model_versions[0]
        
        logger.info(f"✅ Found {len(model_versions)} model versions. Loading latest: {latest_file} (v{latest_version})")
        
        # Load the latest model
        if self._load_model(latest_file):
            self.current_version = latest_version
            
            # Clean up old models
            self._cleanup_old_models(model_versions)
            return True
        else:
            logger.error("❌ Failed to load latest model. Trying previous version...")
            
            # Try to load the previous version if available
            if len(model_versions) > 1:
                prev_version, prev_file = model_versions[1]
                if self._load_model(prev_file):
                    self.current_version = prev_version
                    self._cleanup_old_models(model_versions)
                    return True
            
            logger.error("❌ All models failed to load. Using fallback initialization.")
            self._initialize_fallback_model()
            return False
    
    def _initialize_fallback_model(self):
        """Initialize a basic model when no saved models are available"""
        logger.info("🔧 Initializing fallback model...")
        
        # Build the model architecture
        self._build_model()
        
        # Set some reasonable defaults
        self.multiplier_range = 0.20
        self.best_accuracy = 75.0
        self.current_version = 0
        
        logger.warning("⚠️ Using fallback model. Accuracy and range may not be optimal.")
    
    def _load_model(self, model_path):
        """Load model with optimized memory usage"""
        start_time = time.time()
        logger.info(f"🔄 Loading model from {model_path}...")
        
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            # Extract model components
            self.multiplier_range = model_data.get("multiplier_range", 0.20)
            self.best_accuracy = model_data.get("best_accuracy", 75.0)
            
            # Build model architecture (must match training)
            self._build_model()
            
            # Set weights (optimized for fast loading)
            self.model.set_weights(model_data["model_weights"])
            
            # Clear memory after loading
            tf.keras.backend.clear_session()
            gc.collect()
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f} seconds | "
                        f"Accuracy: {self.best_accuracy:.2f}% | "
                        f"Range: {self.multiplier_range:.2%} | "
                        f"Version: {self.current_version}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Model load failed: {str(e)}")
            return False
    
    def _build_model(self):
        """Build model with optimized architecture for fast inference"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
        from tensorflow.keras.regularizers import l2
        
        # Build lightweight model (optimized for inference speed)
        input_layer = Input(shape=(self.sequence_length, 1))
        
        # Simplified architecture for faster inference
        x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        x = MaxPooling1D(2)(x)
        x = Bidirectional(LSTM(64, kernel_regularizer=l2(0.001)))(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=input_layer, outputs=output)
        
        # Compile with inference-focused settings
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    def _cleanup_old_models(self, model_versions):
        """Delete old model files to save space"""
        if len(model_versions) <= MAX_MODELS_TO_KEEP:
            logger.info("🧹 No old models to clean up")
            return
        
        # Keep only the latest model
        models_to_delete = model_versions[MAX_MODELS_TO_KEEP:]
        
        for version, file_path in models_to_delete:
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Deleted old model: {file_path} (v{version})")
            except Exception as e:
                logger.error(f"❌ Failed to delete {file_path}: {str(e)}")
        
        # Verify cleanup
        remaining_models = glob.glob(f"{MODEL_PREFIX}*[0-9]{MODEL_EXTENSION}")
        logger.info(f"💾 {len(remaining_models)} model(s) remaining after cleanup")
    
    def predict(self, data):
        """Make fast prediction with optimized processing"""
        global last_prediction_time
        last_prediction_time = datetime.now()
        
        # Validate input
        if not isinstance(data, list) or len(data) != self.sequence_length:
            return {
                "error": "Invalid input",
                "details": f"Expected {self.sequence_length} values, got {len(data)}"
            }
        
        try:
            # Preprocess data for model input
            processed_data = np.array(data).reshape(1, self.sequence_length, 1)
            
            # Make prediction (optimized for speed)
            start_time = time.time()
            raw_prediction = self.model.predict(processed_data, verbose=0)[0][0]
            prediction_time = time.time() - start_time
            
            # Calculate dynamic range based on accuracy
            range_adjustment = (self.best_accuracy - 75.0) / 100
            current_range = min(0.65, max(0.05, self.multiplier_range * (1 + range_adjustment * 0.8)))
            
            # Calculate cash-out range
            cash_out_range = {
                "lower": max(1.0, raw_prediction * (1 - current_range)),
                "upper": raw_prediction * (1 + current_range),
                "range_percentage": current_range * 100
            }
            
            return {
                "prediction": float(raw_prediction),
                "cash_out_range": cash_out_range,
                "prediction_time_ms": prediction_time * 1000,
                "accuracy": self.best_accuracy,
                "range": current_range,
                "model_version": self.current_version
            }
        except Exception as e:
            return {"error": str(e)}

def circuit_breaker(func):
    """Circuit breaker decorator to prevent repeated failed warm-ups"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global circuit_open, circuit_open_time, circuit_failure_count
        
        # Check if circuit is open and needs to be reset
        if circuit_open and circuit_open_time and (datetime.now() - circuit_open_time) > timedelta(seconds=CIRCUIT_BREAKER_RESET_TIME):
            logger.info("🔄 Circuit breaker reset after timeout")
            circuit_open = False
            circuit_failure_count = 0
        
        # If circuit is open, skip execution
        if circuit_open:
            logger.warning("🚫 Circuit breaker open - skipping warm-up")
            return False
        
        try:
            result = func(*args, **kwargs)
            # Reset failure count on success
            circuit_failure_count = 0
            return result
        except Exception as e:
            # Increment failure count
            circuit_failure_count += 1
            logger.error(f"🔥 Warm-up failed ({circuit_failure_count}/{CIRCUIT_BREAKER_THRESHOLD}): {str(e)}")
            
            # Open circuit if threshold reached
            if circuit_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                circuit_open = True
                circuit_open_time = datetime.now()
                logger.critical("💥 CIRCUIT BREAKER OPENED - Too many warm-up failures")
            
            raise
    
    return wrapper

def measure_performance(func):
    """Decorator to measure and log warm-up performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record warm-up metrics
            warmup_history.append({
                "timestamp": datetime.now(),
                "duration": duration,
                "success": True,
                "model_version": predictor.current_version if predictor else None
            })
            
            # Trim history to max size
            if len(warmup_history) > MAX_HISTORY_SIZE:
                warmup_history.pop(0)
                
            return result
        except Exception as e:
            duration = time.time() - start_time
            warmup_history.append({
                "timestamp": datetime.now(),
                "duration": duration,
                "success": False,
                "error": str(e)
            })
            raise
    
    return wrapper

def get_warmup_strategy():
    """Determine the appropriate warm-up strategy based on current conditions"""
    # Check memory usage
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent()
    
    # Check model complexity
    model_complexity = 1.0  # Could be based on model size or architecture
    
    # Check time of day (for potential traffic patterns)
    hour = datetime.now().hour
    is_peak_time = 7 <= hour < 23  # 7 AM to 11 PM
    
    # Determine strategy
    if memory_percent > 70 or not is_peak_time:
        return "light"
    elif memory_percent > 40 or model_complexity > 0.7:
        return "medium"
    else:
        return "heavy"

@measure_performance
@circuit_breaker
def advanced_warmup_model(strategy="auto"):
    """Advanced warm-up with multiple strategies and fallbacks"""
    global last_prediction_time
    
    start_time = time.time()
    logger.info("🔥 Starting ADVANCED warm-up sequence...")
    
    # Determine strategy if auto
    if strategy == "auto":
        strategy = get_warmup_strategy()
        logger.info(f"🧠 Selected warm-up strategy: {strategy}")
    
    try:
        # Basic warm-up (always needed)
        dummy_data = list(np.random.uniform(1.0, 10.0, predictor.sequence_length))
        
        # Different warm-up strategies based on needs
        if strategy == "light":
            logger.info("💡 Light warm-up: Single prediction")
            result = predictor.predict(dummy_data)
            if "error" in result:
                raise ValueError(f"Light warm-up failed: {result['error']}")
                
        elif strategy == "medium":
            logger.info("💡 Medium warm-up: Multiple predictions with varied inputs")
            # First prediction (cold start)
            result = predictor.predict(dummy_data)
            if "error" in result:
                raise ValueError(f"Medium warm-up failed on first prediction: {result['error']}")
            
            # Second prediction with different pattern (simulates real usage)
            dummy_data2 = [x * 1.2 for x in dummy_data]
            result2 = predictor.predict(dummy_data2)
            if "error" in result2:
                raise ValueError(f"Medium warm-up failed on second prediction: {result2['error']}")
                
        else:  # heavy
            logger.info("💡 Heavy warm-up: Stress test with multiple patterns")
            # Run through several prediction patterns
            results = []
            for i in range(5):
                # Generate different patterns
                if i == 0:
                    pattern = dummy_data  # Random
                elif i == 1:
                    pattern = [x * 1.1 for x in dummy_data]  # Upward trend
                elif i == 2:
                    pattern = [x * 0.9 for x in dummy_data]  # Downward trend
                elif i == 3:
                    pattern = dummy_data[::-1]  # Reversed
                else:
                    pattern = [x + (0.5 * np.sin(i)) for x in dummy_data]  # Oscillating
                
                result = predictor.predict(pattern)
                if "error" in result:
                    logger.warning(f"Heavy warm-up failed on pattern {i}: {result['error']}")
                results.append(result)
            
            # Verify at least 3 patterns succeeded
            success_count = sum(1 for r in results if "error" not in r)
            if success_count < 3:
                raise ValueError(f"Heavy warm-up failed: {success_count}/5 patterns succeeded")
        
        # Record success
        prediction_time = time.time() - start_time
        last_prediction_time = datetime.now()
        
        # Log detailed metrics
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"🔥 Warm-up SUCCESS! Strategy: {strategy} | "
                    f"Time: {prediction_time:.2f}s | "
                    f"Memory: {memory_usage:.2f}MB | "
                    f"Model version: {predictor.current_version}")
        
        return True
        
    except Exception as e:
        logger.error(f"🔥 Warm-up FAILED: {str(e)}")
        raise

def safe_warmup(retry_count=0):
    """Safely execute warm-up with retries and backoff"""
    try:
        # Check if we can even attempt warm-up
        if not predictor or not predictor.model:
            logger.warning("⚠️ Cannot warm up: Predictor not initialized")
            return False
            
        # Try warm-up with the current strategy
        return advanced_warmup_model()
        
    except Exception as e:
        # Handle retry logic
        if retry_count < WARMUP_RETRIES:
            backoff_time = WARMUP_TIMEOUT * (WARMUP_BACKOFF_FACTOR ** retry_count)
            logger.warning(f"⚠️ Warm-up failed (attempt {retry_count+1}/{WARMUP_RETRIES}), "
                          f"retrying in {backoff_time:.1f}s: {str(e)}")
            time.sleep(backoff_time)
            return safe_warmup(retry_count + 1)
        else:
            logger.error("❌ Warm-up failed after maximum retries")
            return False

def get_dynamic_warmup_interval():
    """Calculate dynamic warm-up interval based on historical performance"""
    if not warmup_history:
        return WARMUP_INTERVAL
    
    # Analyze recent warm-up success rate
    recent = warmup_history[-min(20, len(warmup_history)):]
    success_rate = sum(1 for entry in recent if entry["success"]) / len(recent)
    
    # If success rate is high, we can increase interval slightly
    if success_rate > 0.95:
        return min(MAX_WARMUP_INTERVAL, WARMUP_INTERVAL * 1.1)
    # If success rate is low, decrease interval to be safer
    elif success_rate < 0.8:
        return max(MIN_WARMUP_INTERVAL, WARMUP_INTERVAL * 0.85)
    else:
        return WARMUP_INTERVAL

def keep_alive_worker():
    """Advanced keep-alive worker with dynamic interval and circuit breaker"""
    global keep_alive_active
    
    logger.info("🛡️ Starting ADVANCED keep-alive system...")
    
    while keep_alive_active:
        try:
            # Calculate dynamic interval with jitter
            base_interval = get_dynamic_warmup_interval()
            jitter = random.uniform(-WARMUP_JITTER/2, WARMUP_JITTER/2)
            current_interval = max(MIN_WARMUP_INTERVAL, min(MAX_WARMUP_INTERVAL, base_interval + jitter))
            
            logger.debug(f"⏱️ Next warm-up in {current_interval:.1f} seconds "
                         f"(base: {base_interval:.1f}s, jitter: {jitter:.1f}s)")
            
            # Sleep for the calculated interval (in smaller chunks for responsiveness)
            slept = 0
            sleep_step = 5
            while slept < current_interval and keep_alive_active:
                time.sleep(min(sleep_step, current_interval - slept))
                slept += sleep_step
            
            # Skip if keep_alive was deactivated during sleep
            if not keep_alive_active:
                break
                
            # Execute warm-up with queue to prevent overlapping
            try:
                warmup_queue.put_nowait(True)
                safe_warmup()
            except queue.Full:
                logger.debug("⏭️ Warm-up skipped: Previous warm-up still running")
            finally:
                try:
                    warmup_queue.get_nowait()
                    warmup_queue.task_done()
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"🛡️ Keep-alive system error: {str(e)}")
            time.sleep(10)  # Brief pause before retrying

def start_keep_alive_system():
    """Start the advanced keep-alive system"""
    global keep_alive_thread, keep_alive_active
    
    # Stop any existing keep-alive
    stop_keep_alive_system()
    
    # Reset state
    keep_alive_active = True
    
    # Start new keep-alive thread
    keep_alive_thread = threading.Thread(target=keep_alive_worker, daemon=True)
    keep_alive_thread.start()
    logger.info("🛡️ Advanced keep-alive system activated")

def stop_keep_alive_system():
    """Stop the keep-alive system gracefully"""
    global keep_alive_active
    
    if keep_alive_active:
        keep_alive_active = False
        logger.info("🛡️ Shutting down keep-alive system...")

def warmup_analysis():
    """Analyze warm-up history for insights"""
    if not warmup_history:
        return {
            "status": "no_data",
            "message": "No warm-up history available"
        }
    
    # Calculate metrics
    successful = [entry for entry in warmup_history if entry["success"]]
    failed = [entry for entry in warmup_history if not entry["success"]]
    
    # Time-based metrics
    now = datetime.now()
    last_hour = [e for e in warmup_history if now - e["timestamp"] < timedelta(hours=1)]
    
    return {
        "total": len(warmup_history),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(warmup_history) if warmup_history else 0,
        "last_hour_count": len(last_hour),
        "last_hour_success_rate": sum(1 for e in last_hour if e["success"]) / len(last_hour) if last_hour else 0,
        "average_duration": sum(e["duration"] for e in successful) / len(successful) if successful else 0,
        "circuit_breaker_status": "open" if circuit_open else "closed",
        "circuit_breaker_failures": circuit_failure_count,
        "recommended_interval": get_dynamic_warmup_interval()
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with cold start prevention"""
    data = request.json
    
    if not data or 'values' not in data:
        return jsonify({
            "error": "Invalid request",
            "details": "Request must include 'values' array"
        }), 400
    
    values = data['values']
    
    # Validate input length
    if not isinstance(values, list) or len(values) != predictor.sequence_length:
        return jsonify({
            "error": "Invalid input",
            "details": f"Expected {predictor.sequence_length} values, got {len(values)}"
        }), 400
    
    # Convert to float
    try:
        input_data = [float(x) for x in values]
    except (TypeError, ValueError) as e:
        return jsonify({
            "error": "Invalid input format",
            "details": "All values must be numerical"
        }), 400
    
    # Make prediction
    result = predictor.predict(input_data)
    
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check with model version info"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": predictor.current_version if predictor else None,
        "best_accuracy": predictor.best_accuracy if predictor else 0,
        "multiplier_range": predictor.multiplier_range if predictor else 0.20,
        "last_prediction": last_prediction_time.isoformat() if last_prediction_time else None,
        "memory_rss_mb": memory_info.rss / 1024 / 1024,
        "memory_vms_mb": memory_info.vms / 1024 / 1024,
        "models_remaining": len(glob.glob(f"{MODEL_PREFIX}*[0-9]{MODEL_EXTENSION}")),
        "circuit_breaker_status": "open" if circuit_open else "closed",
        "circuit_breaker_failures": circuit_failure_count
    })

@app.route('/warmup', methods=['GET'])
def warmup():
    """Manual warmup endpoint (legacy)"""
    success = safe_warmup()
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Model is ready for instant predictions" if success else "Warm-up failed",
        "model_version": predictor.current_version if predictor else None
    })

@app.route('/warmup/advanced', methods=['GET'])
def advanced_warmup():
    """Advanced warm-up endpoint with strategy selection"""
    strategy = request.args.get('strategy', 'auto')
    if strategy not in ['auto', 'light', 'medium', 'heavy']:
        return jsonify({
            "error": "Invalid strategy",
            "details": "Strategy must be 'auto', 'light', 'medium', or 'heavy'"
        }), 400
    
    try:
        success = safe_warmup()
        return jsonify({
            "status": "success" if success else "failed",
            "strategy": strategy,
            "analysis": warmup_analysis()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "analysis": warmup_analysis()
        }), 500

@app.route('/warmup/status', methods=['GET'])
def warmup_status():
    """Get warm-up system status"""
    return jsonify({
        "keep_alive_active": keep_alive_active,
        "circuit_breaker_open": circuit_open,
        "circuit_breaker_failures": circuit_failure_count,
        "next_warmup_in": get_dynamic_warmup_interval(),
        "warmup_history_size": len(warmup_history),
        "analysis": warmup_analysis()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    model_files = glob.glob(f"{MODEL_PREFIX}*[0-9]{MODEL_EXTENSION}")
    model_versions = []
    
    for file in model_files:
        try:
            version = int(re.search(rf"{MODEL_PREFIX}(\d+){MODEL_EXTENSION}", file).group(1))
            model_versions.append({
                "file": file,
                "version": version,
                "size_kb": round(os.path.getsize(file) / 1024, 2)
            })
        except (AttributeError, ValueError):
            continue
    
    model_versions.sort(key=lambda x: x["version"], reverse=True)
    
    return jsonify({
        "total_models": len(model_versions),
        "models": model_versions,
        "active_model": predictor.current_version if predictor else None,
        "analysis": warmup_analysis()
    })

def start_advanced_keep_alive():
    """Initialize the advanced keep-alive system"""
    # Warm up immediately
    safe_warmup()
    
    # Start the keep-alive system
    start_keep_alive_system()
    
    # Log warm-up metrics periodically
    def log_metrics():
        while keep_alive_active:
            time.sleep(300)  # Every 5 minutes
            analysis = warmup_analysis()
            logger.info(f"📊 Warm-up metrics - Success rate: {analysis['success_rate']:.2%} | "
                        f"Total: {analysis['total']} | "
                        f"Last hour: {analysis['last_hour_count']} | "
                        f"Interval: {analysis['recommended_interval']:.1f}s")
    
    metrics_thread = threading.Thread(target=log_metrics, daemon=True)
    metrics_thread.start()

if __name__ == "__main__":
    # Initialize predictor
    logger.info("🚀 Starting Crash Predictor...")
    
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    os.chdir("models")  # Work within models directory
    
    global predictor
    predictor = OptimizedPredictor()
    
    # Warm up the model immediately
    safe_warmup()
    
    # Start keep-alive system
    start_advanced_keep_alive()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"/WebAPI running on port {port} - ready for instant predictions!")
    app.run(host='0.0.0.0', port=port)
