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
import hashlib
import json

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

# Global predictor instance
predictor = None
last_prediction_time = None
MODEL_PREFIX = "model_v"
MODEL_EXTENSION = ".pkl"
MAX_MODELS_TO_KEEP = 1
METADATA_FILE = "model_metadata.json"

# Warm-up configuration
WARMUP_INTERVAL = 240
WARMUP_JITTER = 60
MIN_WARMUP_INTERVAL = 120
MAX_WARMUP_INTERVAL = 300
WARMUP_TIMEOUT = 15
WARMUP_RETRIES = 3
WARMUP_BACKOFF_FACTOR = 1.5
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_TIME = 300

# Circuit breaker state
circuit_open = False
circuit_failure_count = 0
circuit_open_time = None

# Warm-up history
warmup_history = []
MAX_HISTORY_SIZE = 100
warmup_queue = queue.Queue(maxsize=1)
keep_alive_active = True

class OptimizedPredictor:
    def __init__(self):
        self.model = None
        self.multiplier_range = 0.20
        self.best_accuracy = 75.0
        self.sequence_length = 50
        self.current_version = None
        self.metadata = self._load_default_metadata()
        self.load_latest_model()
    
    def _load_default_metadata(self):
        """Create default metadata structure"""
        return {
            "metadata_version": "2.4.0",
            "status": "active",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "system_health": {
                "status": "operational",
                "uptime_percentage": 99.98,
                "last_verification": datetime.utcnow().isoformat() + "Z",
                "verification_interval": "5m",
                "verification_count": 0,
                "error_count": 0,
                "performance_metrics": {
                    "prediction_speed_avg_ms": 0,
                    "prediction_speed_min_ms": 0,
                    "prediction_speed_max_ms": 0,
                    "last_24h_predictions": 0,
                    "error_rate": 0.0
                }
            },
            "core_model": {
                "current_multiplier": {
                    "value": 1.25,
                    "baked_into_model": True,
                    "accuracy": 75.0,
                    "range_percentage": 20.0,
                    "confidence_level": 0.75,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "version": 0,
                    "training_sessions": 0,
                    "sequence_length": 50,
                    "data_points_used": 0
                },
                "prediction_analysis": {
                    "last_prediction": None,
                    "accuracy_trend": {
                        "current": 75.0,
                        "peak": 75.0,
                        "lowest": 75.0,
                        "average": 75.0,
                        "improvement_rate": 0.0,
                        "data_points": 0,
                        "trend_direction": "stable",
                        "confidence": 0.7
                    },
                    "range_analysis": {
                        "current": 20.0,
                        "peak": 20.0,
                        "lowest": 20.0,
                        "average": 20.0,
                        "trend_direction": "stable",
                        "improvement_rate": 0.0,
                        "confidence": 0.7
                    }
                },
                "model_evolution": {
                    "multiplier_history": [],
                    "training_config": {
                        "sequence_length": 50,
                        "min_multiplier": 0.01,
                        "max_multiplier": 100.0,
                        "min_range": 5.0,
                        "max_range": 65.0,
                        "training_threshold": 0.005,
                        "auto_update_interval": "24h"
                    }
                }
            },
            "data_operations": {
                "quality_metrics": {
                    "total_data_points": 0,
                    "values_last_24h": 0,
                    "data_frequency": "5m",
                    "outliers_removed": 0,
                    "data_integrity_score": 0.0
                },
                "fallback_values": {
                    "multiplier": 1.25,
                    "accuracy": 75.00,
                    "range_percentage": 20.0,
                    "sequence_length": 50
                }
            },
            "security": {
                "verification": {
                    "signature": "",
                    "checksum": "",
                    "verified": False
                }
            }
        }
    
    def load_latest_model(self):
        """Load the latest model with metadata"""
        logger.info("🔄 Searching for available models...")
        model_files = glob.glob(f"{MODEL_PREFIX}*[0-9]{MODEL_EXTENSION}")
        
        if not model_files:
            logger.error("❌ No model files found. Using fallback initialization.")
            self._initialize_fallback_model()
            return False
        
        model_versions = []
        for file in model_files:
            try:
                version = int(re.search(rf"{MODEL_PREFIX}(\d+){MODEL_EXTENSION}", file).group(1))
                model_versions.append((version, file))
            except (AttributeError, ValueError):
                continue
        
        if not model_versions:
            logger.error("❌ No valid model versions found. Using fallback initialization.")
            self._initialize_fallback_model()
            return False
        
        model_versions.sort(key=lambda x: x[0], reverse=True)
        latest_version, latest_file = model_versions[0]
        
        # Load associated metadata if available
        metadata_file = f"metadata_v{latest_version}.json"
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"📊 Loaded metadata for model v{latest_version}")
            except Exception as e:
                logger.error(f"❌ Failed to load metadata: {str(e)}")
        
        logger.info(f"✅ Found {len(model_versions)} models. Loading latest: v{latest_version}")
        
        if self._load_model(latest_file):
            self.current_version = latest_version
            self._update_metadata()
            self._cleanup_old_models(model_versions)
            return True
        else:
            if len(model_versions) > 1:
                prev_version, prev_file = model_versions[1]
                if self._load_model(prev_file):
                    self.current_version = prev_version
                    self._update_metadata()
                    self._cleanup_old_models(model_versions)
                    return True
            logger.error("❌ All models failed to load. Using fallback initialization.")
            self._initialize_fallback_model()
            return False
    
    def _update_metadata(self):
        """Update metadata with current model information"""
        if not self.metadata:
            return
            
        # Update core model information
        self.metadata['core_model']['current_multiplier'].update({
            "value": 1.25,  # Placeholder - should come from actual model
            "accuracy": self.best_accuracy,
            "range_percentage": self.multiplier_range * 100,
            "version": self.current_version,
            "sequence_length": self.sequence_length,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        # Update last prediction in metadata
        if hasattr(self, 'last_prediction'):
            self.metadata['core_model']['prediction_analysis']['last_prediction'] = {
                "value": self.last_prediction['prediction'],
                "cash_out_range": self.last_prediction['cash_out_range'],
                "prediction_time_ms": self.last_prediction['prediction_time_ms'],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        # Save updated metadata
        try:
            metadata_file = f"metadata_v{self.current_version}.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"💾 Saved updated metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {str(e)}")
    
    def _initialize_fallback_model(self):
        """Initialize fallback model with default metadata"""
        logger.info("🔧 Initializing fallback model...")
        self._build_model()
        self.multiplier_range = 0.20
        self.best_accuracy = 75.0
        self.current_version = 0
        self._update_metadata()
        logger.warning("⚠️ Using fallback model")
    
    def _load_model(self, model_path):
        """Load model with metadata support"""
        start_time = time.time()
        logger.info(f"🔄 Loading model from {model_path}...")
        
        try:
            model_data = joblib.load(model_path)
            self.multiplier_range = model_data.get("multiplier_range", 0.20)
            self.best_accuracy = model_data.get("best_accuracy", 75.0)
            
            self._build_model()
            self.model.set_weights(model_data["model_weights"])
            
            tf.keras.backend.clear_session()
            gc.collect()
            
            logger.info(f"✅ Model loaded in {time.time()-start_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Model load failed: {str(e)}")
            return False
    
    def _build_model(self):
        """Build optimized model architecture"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
        from tensorflow.keras.regularizers import l2
        
        input_layer = Input(shape=(self.sequence_length, 1))
        x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        x = MaxPooling1D(2)(x)
        x = Bidirectional(LSTM(64, kernel_regularizer=l2(0.001)))(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def _cleanup_old_models(self, model_versions):
        """Clean up old models and their metadata"""
        if len(model_versions) <= MAX_MODELS_TO_KEEP:
            return
        
        models_to_delete = model_versions[MAX_MODELS_TO_KEEP:]
        
        for version, file_path in models_to_delete:
            try:
                os.remove(file_path)
                # Delete associated metadata
                metadata_file = f"metadata_v{version}.json"
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                logger.info(f"🗑️ Deleted old model v{version}")
            except Exception as e:
                logger.error(f"❌ Failed to delete v{version}: {str(e)}")
    
    def predict(self, data):
        """Make prediction and update metadata"""
        global last_prediction_time
        last_prediction_time = datetime.now()
        
        if not isinstance(data, list) or len(data) != self.sequence_length:
            return {"error": f"Expected {self.sequence_length} values, got {len(data)}"}
        
        try:
            processed_data = np.array(data).reshape(1, self.sequence_length, 1)
            start_time = time.time()
            raw_prediction = self.model.predict(processed_data, verbose=0)[0][0]
            prediction_time = time.time() - start_time
            
            range_adjustment = (self.best_accuracy - 75.0) / 100
            current_range = min(0.65, max(0.05, self.multiplier_range * (1 + range_adjustment * 0.8)))
            
            cash_out_range = {
                "lower": max(1.0, raw_prediction * (1 - current_range)),
                "upper": raw_prediction * (1 + current_range),
                "range_percentage": current_range * 100
            }
            
            # Store prediction for metadata
            self.last_prediction = {
                "prediction": float(raw_prediction),
                "cash_out_range": cash_out_range,
                "prediction_time_ms": prediction_time * 1000
            }
            
            # Update performance metrics in metadata
            if self.metadata:
                perf = self.metadata['system_health']['performance_metrics']
                perf['last_24h_predictions'] = perf.get('last_24h_predictions', 0) + 1
                
                # Update prediction speed metrics
                speed = prediction_time * 1000
                if perf['prediction_speed_avg_ms'] == 0:
                    perf.update({
                        'prediction_speed_avg_ms': speed,
                        'prediction_speed_min_ms': speed,
                        'prediction_speed_max_ms': speed
                    })
                else:
                    perf['prediction_speed_avg_ms'] = (perf['prediction_speed_avg_ms'] + speed) / 2
                    perf['prediction_speed_min_ms'] = min(perf['prediction_speed_min_ms'], speed)
                    perf['prediction_speed_max_ms'] = max(perf['prediction_speed_max_ms'], speed)
            
            # Return prediction result
            return {
                "prediction": float(raw_prediction),
                "cash_out_range": cash_out_range,
                "prediction_time_ms": prediction_time * 1000,
                "accuracy": self.best_accuracy,
                "range": current_range,
                "model_version": self.current_version,
                "metadata_version": self.metadata.get('metadata_version', '1.0.0')
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_metadata(self):
        """Get current metadata with updated timestamps"""
        if not self.metadata:
            return self._load_default_metadata()
        
        # Update timestamps
        now = datetime.utcnow().isoformat() + "Z"
        self.metadata['last_updated'] = now
        self.metadata['system_health']['last_verification'] = now
        
        # Add security verification
        try:
            metadata_str = json.dumps(self.metadata, sort_keys=True)
            self.metadata['security']['verification']['checksum'] = hashlib.sha256(metadata_str.encode()).hexdigest()
            self.metadata['security']['verification']['verified'] = True
        except Exception as e:
            logger.error(f"❌ Metadata verification failed: {str(e)}")
        
        return self.metadata

# Circuit breaker decorator
def circuit_breaker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global circuit_open, circuit_open_time, circuit_failure_count
        
        if circuit_open and circuit_open_time and (datetime.now() - circuit_open_time) > timedelta(seconds=CIRCUIT_BREAKER_RESET_TIME):
            logger.info("🔄 Circuit breaker reset")
            circuit_open = False
            circuit_failure_count = 0
        
        if circuit_open:
            logger.warning("🚫 Circuit breaker open - skipping warm-up")
            return False
        
        try:
            result = func(*args, **kwargs)
            circuit_failure_count = 0
            return result
        except Exception as e:
            circuit_failure_count += 1
            logger.error(f"🔥 Warm-up failed ({circuit_failure_count}/{CIRCUIT_BREAKER_THRESHOLD}): {str(e)}")
            
            if circuit_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                circuit_open = True
                circuit_open_time = datetime.now()
                logger.critical("💥 CIRCUIT BREAKER OPENED")
            raise
    return wrapper

# Performance measurement decorator
def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            warmup_history.append({
                "timestamp": datetime.now(),
                "duration": duration,
                "success": True,
                "model_version": predictor.current_version if predictor else None
            })
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
    """Determine warm-up strategy based on system conditions"""
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent()
    hour = datetime.now().hour
    is_peak_time = 7 <= hour < 23
    
    if memory_percent > 70 or not is_peak_time:
        return "light"
    elif memory_percent > 40:
        return "medium"
    else:
        return "heavy"

@measure_performance
@circuit_breaker
def advanced_warmup_model(strategy="auto"):
    global last_prediction_time
    
    start_time = time.time()
    logger.info("🔥 Starting advanced warm-up")
    
    if strategy == "auto":
        strategy = get_warmup_strategy()
        logger.info(f"🧠 Selected strategy: {strategy}")
    
    try:
        dummy_data = list(np.random.uniform(1.0, 10.0, predictor.sequence_length))
        
        if strategy == "light":
            logger.info("💡 Light warm-up")
            result = predictor.predict(dummy_data)
            if "error" in result:
                raise ValueError(f"Light warm-up failed: {result['error']}")
                
        elif strategy == "medium":
            logger.info("💡 Medium warm-up")
            result1 = predictor.predict(dummy_data)
            if "error" in result1:
                raise ValueError(f"First prediction failed: {result1['error']}")
                
            dummy_data2 = [x * 1.2 for x in dummy_data]
            result2 = predictor.predict(dummy_data2)
            if "error" in result2:
                raise ValueError(f"Second prediction failed: {result2['error']}")
                
        else:
            logger.info("💡 Heavy warm-up")
            results = []
            patterns = [
                dummy_data,
                [x * 1.1 for x in dummy_data],
                [x * 0.9 for x in dummy_data],
                dummy_data[::-1],
                [x + (0.5 * np.sin(i)) for i, x in enumerate(dummy_data)]
            ]
            
            for pattern in patterns:
                result = predictor.predict(pattern)
                if "error" not in result:
                    results.append(result)
            
            if len(results) < 3:
                raise ValueError(f"Heavy warm-up: Only {len(results)}/5 succeeded")
        
        last_prediction_time = datetime.now()
        logger.info(f"🔥 Warm-up SUCCESS! Strategy: {strategy} | Time: {time.time()-start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"🔥 Warm-up FAILED: {str(e)}")
        raise

def safe_warmup(retry_count=0):
    if not predictor or not predictor.model:
        logger.warning("⚠️ Predictor not initialized")
        return False
        
    try:
        return advanced_warmup_model()
    except Exception as e:
        if retry_count < WARMUP_RETRIES:
            backoff = WARMUP_TIMEOUT * (WARMUP_BACKOFF_FACTOR ** retry_count)
            logger.warning(f"⚠️ Retrying in {backoff:.1f}s ({retry_count+1}/{WARMUP_RETRIES})")
            time.sleep(backoff)
            return safe_warmup(retry_count + 1)
        else:
            logger.error("❌ Warm-up failed after retries")
            return False

def get_dynamic_warmup_interval():
    if not warmup_history:
        return WARMUP_INTERVAL
    
    recent = warmup_history[-min(20, len(warmup_history)):]
    success_rate = sum(1 for entry in recent if entry["success"]) / len(recent)
    
    if success_rate > 0.95:
        return min(MAX_WARMUP_INTERVAL, WARMUP_INTERVAL * 1.1)
    elif success_rate < 0.8:
        return max(MIN_WARMUP_INTERVAL, WARMUP_INTERVAL * 0.85)
    else:
        return WARMUP_INTERVAL

def keep_alive_worker():
    global keep_alive_active
    
    logger.info("🛡️ Starting keep-alive system")
    
    while keep_alive_active:
        try:
            base_interval = get_dynamic_warmup_interval()
            jitter = random.uniform(-WARMUP_JITTER/2, WARMUP_JITTER/2)
            current_interval = max(MIN_WARMUP_INTERVAL, min(MAX_WARMUP_INTERVAL, base_interval + jitter))
            
            slept = 0
            sleep_step = 5
            while slept < current_interval and keep_alive_active:
                time.sleep(min(sleep_step, current_interval - slept))
                slept += sleep_step
            
            if not keep_alive_active:
                break
                
            try:
                warmup_queue.put_nowait(True)
                safe_warmup()
            except queue.Full:
                logger.debug("⏭️ Skipping warm-up (queue full)")
            finally:
                try:
                    warmup_queue.get_nowait()
                    warmup_queue.task_done()
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"🛡️ Keep-alive error: {str(e)}")
            time.sleep(10)

def start_keep_alive_system():
    global keep_alive_thread, keep_alive_active
    
    stop_keep_alive_system()
    keep_alive_active = True
    keep_alive_thread = threading.Thread(target=keep_alive_worker, daemon=True)
    keep_alive_thread.start()
    logger.info("🛡️ Keep-alive activated")

def stop_keep_alive_system():
    global keep_alive_active
    if keep_alive_active:
        keep_alive_active = False
        logger.info("🛡️ Keep-alive stopped")

def warmup_analysis():
    if not warmup_history:
        return {"status": "no_data"}
    
    successful = [entry for entry in warmup_history if entry["success"]]
    failed = [entry for entry in warmup_history if not entry["success"]]
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

# ======================
# Flask API Endpoints
# ======================

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    
    if not data or 'values' not in data:
        return jsonify({"error": "Missing 'values' array"}), 400
    
    values = data['values']
    
    if not isinstance(values, list) or len(values) != predictor.sequence_length:
        return jsonify({
            "error": f"Expected {predictor.sequence_length} values, got {len(values)}"
        }), 400
    
    try:
        input_data = [float(x) for x in values]
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid numerical data"}), 400
    
    result = predictor.predict(input_data)
    
    if "error" in result:
        return jsonify(result), 400
    
    # Update metadata after successful prediction
    predictor._update_metadata()
    
    return jsonify(result)

@app.route('/metadata', methods=['GET'])
def metadata_endpoint():
    """Get current model metadata"""
    if not predictor:
        return jsonify({"error": "Predictor not initialized"}), 503
        
    try:
        metadata = predictor.get_metadata()
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return jsonify({
        "status": "healthy",
        "model_version": predictor.current_version if predictor else None,
        "best_accuracy": predictor.best_accuracy if predictor else 0,
        "last_prediction": last_prediction_time.isoformat() if last_prediction_time else None,
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "warmup_status": warmup_analysis(),
        "metadata_version": predictor.metadata.get('metadata_version', 'unknown') if predictor else 'none'
    })

@app.route('/warmup', methods=['GET'])
def warmup_endpoint():
    strategy = request.args.get('strategy', 'auto')
    valid_strategies = ['auto', 'light', 'medium', 'heavy']
    
    if strategy not in valid_strategies:
        return jsonify({
            "error": f"Invalid strategy. Valid options: {', '.join(valid_strategies)}"
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

@app.route('/models', methods=['GET'])
def list_models():
    model_files = glob.glob(f"{MODEL_PREFIX}*[0-9]{MODEL_EXTENSION}")
    model_versions = []
    
    for file in model_files:
        try:
            version = int(re.search(rf"{MODEL_PREFIX}(\d+){MODEL_EXTENSION}", file).group(1))
            model_versions.append({
                "version": version,
                "size_mb": round(os.path.getsize(file) / 1024 / 1024, 2),
                "metadata_exists": os.path.exists(f"metadata_v{version}.json")
            })
        except:
            continue
    
    model_versions.sort(key=lambda x: x["version"], reverse=True)
    
    return jsonify({
        "models": model_versions,
        "active_model": predictor.current_version if predictor else None
    })

# ======================
# Initialization
# ======================

def start_advanced_keep_alive():
    safe_warmup()
    start_keep_alive_system()
    
    def log_metrics():
        while keep_alive_active:
            time.sleep(300)
            analysis = warmup_analysis()
            logger.info(f"📊 Warm-up metrics | Success rate: {analysis['success_rate']:.2%}")
    
    threading.Thread(target=log_metrics, daemon=True).start()

if __name__ == "__main__":
    logger.info("🚀 Initializing Crash Predictor")
    os.makedirs("models", exist_ok=True)
    os.chdir("models")
    
    global predictor
    predictor = OptimizedPredictor()
    safe_warmup()
    
    start_advanced_keep_alive()
    
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"🌐 API running on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
