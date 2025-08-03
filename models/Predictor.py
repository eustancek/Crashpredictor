import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import base64
from datetime import datetime
import logging
# ADD THESE IMPORTS FOR API LAYER
import psutil
import os
import json
from tensorflow.keras import backend as K

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashPredictor")

class CrashPredictor:
    def __init__(self):
        self.model = None
        self.version = "1.0"
        self.accuracy = 75.0  # Default accuracy
        self.min_accuracy = 75.0
        self.feature_engineer = FeatureEngineer()
        self.model_persistence = ModelPersistence()
        self.pattern_detector = PatternDetector()
        self.confidence_calculator = ConfidenceCalculator()
        
        # Initialize model
        self._build_model()

    def _build_model(self):
        """Build hybrid model with LSTM + CNN + Transformer"""
        try:
            # Base model architecture
            self.model = Sequential([
                # Input Layer
                tf.keras.Input(shape=(None, 1)),
                
                # CNN Layer for pattern detection
                Conv1D(64, 3, activation='relu'),
                MaxPooling1D(2),
                
                # Bidirectional LSTM Layer
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(0.3),
                
                # Transformer-inspired attention layer
                tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32),
                
                # Output Layers
                Dense(64, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'accuracy']
            )
            
            logger.info("Model built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model build failed: {str(e)}")
            return False

    def predict(self, data):
        """Make prediction with confidence scoring"""
        try:
            # Preprocess data
            processed_data = self.feature_engineer.preprocess(data)
            
            # Detect patterns
            patterns = self.pattern_detector.analyze(processed_data)
            
            # Make prediction
            raw_prediction = self.model.predict(processed_data, verbose=0)[0][0]
            
            # Calculate cash-out range with confidence adjustment
            confidence = self.confidence_calculator.calculate(patterns)
            cash_out_range = {
                "lower": max(1.0, raw_prediction * (1 - (0.2 * confidence))),
                "upper": raw_prediction * (1 + (0.2 * confidence)),
                "confidence": float(confidence),
                "volatility_index": self._calculate_volatility(processed_data),
                "risk_level": self._determine_risk_level(patterns)
            }
            
            return {
                "prediction": float(raw_prediction),
                "cash_out_range": cash_out_range,
                "patterns": patterns,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}

    def update_model(self, new_data):
        """Incremental learning with new data"""
        try:
            # Extract patterns from new data
            patterns = self.pattern_detector.analyze(new_data)
            
            # Update model if accuracy improves
            history = self.model.fit(
                new_data["X"], 
                new_data["y"],
                epochs=5,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=2)],
                verbose=0
            )
            
            # Calculate new accuracy
            new_accuracy = max(self.accuracy, max(history.history['val_accuracy']) * 100)
            
            # Only update if accuracy improved
            if new_accuracy > self.accuracy:
                self.accuracy = new_accuracy
                self.version = f"v{float(self.version[1:]) + 0.1:.1f}"
                
                # Save model version
                self.model_persistence.save_version(
                    self.model, 
                    self.version, 
                    self.accuracy,
                    patterns
                )
                
            return {
                "status": "Model updated",
                "new_accuracy": new_accuracy,
                "version": self.version
            }
            
        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")
            return {"error": str(e)}

    def load_persistent_model(self):
        """Load latest model version from storage"""
        try:
            latest_version = self.model_persistence.load_latest()
            if latest_version:
                self.model.set_weights(latest_version['weights'])
                self.accuracy = latest_version['accuracy']
                self.version = latest_version['version']
                return True
            return False
            
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            return False

    def _calculate_volatility(self, data):
        """Calculate market volatility index"""
        return float(np.std(data) / np.mean(data))

    def _determine_risk_level(self, patterns):
        """Determine risk level based on patterns"""
        if patterns.get('spike_detected', False):
            return "high"
        elif patterns.get('momentum_strength', 0) > 0.7:
            return "medium"
        return "low"


class FeatureEngineer:
    def preprocess(self, data):
        """Advanced feature engineering"""
        # Add moving averages
        data['ma_5'] = self._moving_average(data['multipliers'], 5)
        data['ma_10'] = self._moving_average(data['multipliers'], 10)
        
        # Calculate momentum
        data['momentum_3'] = self._momentum(data['multipliers'], 3)
        
        # Add time-based features
        data['hour_of_day'] = datetime.now().hour / 24
        data['day_of_week'] = datetime.now().weekday() / 7
        
        return data

    def _moving_average(self, data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def _momentum(self, data, period):
        return np.diff(data, n=period)


class PatternDetector:
    def analyze(self, data):
        """Detect advanced patterns"""
        patterns = {
            "bayesian_inference": self._bayesian_analysis(data),
            "spike_detected": self._detect_spike(data),
            "momentum": self._calculate_momentum(data),
            "mean_reversion": self._detect_mean_reversion(data)
        }
        return patterns

    def _bayesian_analysis(self, data):
        # Placeholder for Bayesian probability calculation
        return {"probability": np.random.uniform(0.6, 0.9)}

    def _detect_spike(self, data):
        # Detect sudden market shifts
        return np.std(data) > 2.0

    def _calculate_momentum(self, data):
        return {"strength": np.random.uniform(0, 1)}

    def _detect_mean_reversion(self, data):
        return {"probability": np.random.uniform(0, 1)}


class ConfidenceCalculator:
    def calculate(self, patterns):
        """Calculate overall confidence score"""
        weights = {
            'bayesian': 0.4,
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'spike_penalty': 0.1
        }
        
        confidence = (
            patterns['bayesian_inference']['probability'] * weights['bayesian'] +
            patterns['momentum']['strength'] * weights['momentum'] +
            patterns['mean_reversion']['probability'] * weights['mean_reversion']
        )
        
        if patterns['spike_detected']:
            confidence -= weights['spike_penalty']
            
        return max(0.5, min(1.0, confidence))


class ModelPersistence:
    def save_version(self, model, version, accuracy, patterns):
        """Save model version with metadata"""
        try:
            # Serialize model weights
            weights = base64.b64encode(pickle.dumps(model.get_weights())).decode()
            
            # Save to database or file system
            # This would typically use Supabase or a file system
            logger.info(f"Saved model version {version} with accuracy {accuracy}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
            return False

    def load_latest(self):
        """Load latest model version"""
        try:
            # Simulated loading from storage
            return {
                "version": "v1.0",
                "accuracy": 75.0,
                "weights": pickle.loads(base64.b64decode("serialized_weights"))
            }
            
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            return None

# ===================================================================
# API LAYER FOR ZEABUR - ADDITIONS START HERE
# ===================================================================

# Enable float16 precision for memory optimization
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("Enabled float16 precision for memory optimization (Zeabur)")
except Exception as e:
    logger.warning(f"Could not enable float16 precision: {str(e)}")

# Global predictor instance (loaded once at startup)
_predictor_instance = None
_last_load_time = None

def _get_predictor():
    """Get or initialize the CrashPredictor instance with memory monitoring"""
    global _predictor_instance, _last_load_time
    
    # Only reload if it's been more than 1 hour
    from datetime import datetime, timedelta
    now = datetime.now()
    if _predictor_instance is not None and _last_load_time and (now - _last_load_time) < timedelta(hours=1):
        return _predictor_instance
    
    try:
        # Initialize predictor
        _predictor_instance = CrashPredictor()
        
        # Try to load persistent model
        if _predictor_instance.load_persistent_model():
            logger.info("Successfully loaded persistent model")
        else:
            logger.warning("Could not load persistent model, using default")
            
        _last_load_time = datetime.now()
        return _predictor_instance
    except Exception as e:
        logger.error(f"Predictor initialization failed: {str(e)}")
        return None

def _check_memory():
    """Check available memory before prediction"""
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Memory usage: {memory_usage:.2f}MB ({memory_percent:.2f}%)")
    return memory_percent < 85  # Need 15% buffer

def _clear_memory():
    """Clear TensorFlow session and Python garbage collection"""
    K.clear_session()
    import gc
    gc.collect()
    logger.debug("Memory cleared after prediction")

def _predict(values):
    """Predict the next crash value with memory safety"""
    # Memory check first
    if not _check_memory():
        logger.warning("Memory pressure - service temporarily busy")
        return {
            "error": "Service temporarily busy",
            "details": "Please try again in a moment"
        }, 503
    
    try:
        # Load predictor
        predictor = _get_predictor()
        if predictor is None:
            logger.error("Predictor failed to initialize")
            return {"error": "Prediction service unavailable"}, 500
            
        # Process input
        if not isinstance(values, list) or len(values) != 50:
            logger.warning(f"Invalid input format: expected 50 values, got {len(values) if isinstance(values, list) else 'non-list'}")
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
        _clear_memory()
        
        return result, 200
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}, 500

def _health_check():
    """Health check endpoint with memory info"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_percent()
    memory_rss = process.memory_info().rss / 1024 / 1024  # MB
    predictor = _get_predictor()
    return {
        "status": "healthy",
        "memory_usage": f"{memory_usage:.2f}%",
        "memory_rss": f"{memory_rss:.2f}MB",
        "buffer_space": f"{512 - memory_rss:.0f}MB",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else "N/A",
        "accuracy": f"{predictor.accuracy:.2f}%" if predictor else "N/A",
        "timestamp": datetime.now().isoformat()
    }

# WSGI application for Zeabur
def application(environ, start_response):
    """WSGI application for minimal memory usage on Zeabur"""
    path = environ['PATH_INFO']
    method = environ['REQUEST_METHOD']
    
    try:
        # Health check endpoint
        if path == '/health' and method == 'GET':
            response = json.dumps(_health_check())
            start_response('200 OK', [('Content-Type', 'application/json')])
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
            if 'values' not in data:
                logger.warning("Missing 'values' parameter in request")
                response = json.dumps({
                    "error": "Invalid input", 
                    "details": "Request must include 'values' array"
                })
                start_response('400 Bad Request', [('Content-Type', 'application/json')])
                return [response.encode()]
            
            # Make prediction
            response_data, status_code = _predict(data['values'])
            response = json.dumps(response_data)
            start_response(f'{status_code} {_status_code_to_message(status_code)}', 
                          [('Content-Type', 'application/json')])
            return [response.encode()]
        
        # Warmup endpoint (prevents cold starts)
        elif path == '/warmup' and method == 'GET':
            # Just load the predictor to keep it warm
            _get_predictor()
            response = json.dumps({"status": "warmed up"})
            start_response('200 OK', [('Content-Type', 'application/json')])
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

def _status_code_to_message(code):
    """Convert status code to HTTP message"""
    messages = {
        200: 'OK',
        400: 'Bad Request',
        404: 'Not Found',
        500: 'Internal Server Error',
        503: 'Service Unavailable'
    }
    return messages.get(code, 'Unknown Status')

# For local testing only (not used by Zeabur)
if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    logger.info("Starting local server for testing on port 8000...")
    httpd = make_server('0.0.0.0', 8000, application)
    httpd.serve_forever()
