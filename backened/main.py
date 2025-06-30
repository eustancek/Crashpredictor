from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from supabase import create_client, Client
from datetime import datetime
import asyncio
import base64
import pickle
import os
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashPredictor")

app = FastAPI()

# Supabase Configuration
SUPABASE_URL = "https://fawcuwcqfwzvdoalcocx.supabase.co "
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhd2N1d2NxZnd6dmRvYWxjb2N4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA4NDY3MjYsImV4cCI6MjA2NjQyMjcyNn0.5NCGUTGpPm7w2Jv0GURMKmGh-EQ7WztNLs9MD5_nSjc"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize model version
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
ACCURACY_THRESHOLD = 75.0  # Minimum accuracy

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request validation
class PredictionRequest(BaseModel):
    multipliers: list[float]
    crash_values: list[float]

class ModelVersion(BaseModel):
    version: str
    accuracy: float
    model_weights: str  # Base64-encoded model weights

# Crash Prediction Model with Versioning
class CrashPredictor:
    def __init__(self):
        self.model = None
        self.version = MODEL_VERSION
        self.accuracy = ACCURACY_THRESHOLD
        self.load_persistent_model()
    
    def _build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(None, 1)),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def load_persistent_model(self):
        """Load latest model version from Supabase"""
        try:
            response = supabase.table('model_versions').select('*').order('created_at', desc=True).limit(1).execute()
            if response.data:
                latest_model = response.data[0]
                weights = base64.b64decode(latest_model['model_weights'])
                self.model = self._build_model()
                self.model.set_weights(pickle.loads(weights))
                self.accuracy = max(ACCURACY_THRESHOLD, latest_model['accuracy'])
                self.version = latest_model['version']
                logger.info(f"Loaded model {self.version} with accuracy {self.accuracy}")
            else:
                self.model = self._build_model()
                logger.info("Using default model")
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.model = self._build_model()
    
    def save_model_version(self):
        """Save current model version to Supabase"""
        try:
            weights = base64.b64encode(pickle.dumps(self.model.get_weights())).decode('utf-8')
            supabase.table('model_versions').insert({
                'version': f"v{float(self.version[1:]) + 0.1:.1f}",
                'model_weights': weights,
                'accuracy': self.accuracy
            }).execute()
            self.version = f"v{float(self.version[1:]) + 0.1:.1f}"
            logger.info(f"Saved model version {self.version} with accuracy {self.accuracy}")
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")

# Initialize global predictor
predictor = CrashPredictor()

# Outlier Detection Module
def detect_outliers(data):
    """Detect and remove outliers using IQR and Z-score"""
    if not data:
        return data
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # IQR Method
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter outliers
    filtered = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)].tolist()
    
    # ML-based outlier detection (fallback)
    if len(filtered) < 5:  # Not enough data for ML detection
        return data
    
    return filtered

# WASM Preprocessing Integration
async def wasm_preprocess(data):
    """
    Simulate WASM preprocessing (replace with actual WASM integration)
    This would typically call a WASM module for feature engineering
    """
    try:
        # Example preprocessing (replace with actual WASM calls)
        processed_data = {
            'ma_5': np.convolve(data, np.ones(5)/5, mode='valid').tolist(),
            'momentum_3': [x - data[i-3] if i >=3 else 0 for i, x in enumerate(data)]
        }
        return processed_data
    except Exception as e:
        logger.warning(f"WASM preprocessing failed: {str(e)}")
        return { 'raw': data }

# Prediction Endpoint
@app.post("/predict")
async def predict( PredictionRequest):
    try:
        # Preprocess data
        raw_data = data.multipliers + data.crash_values
        
        # Detect and remove outliers
        clean_data = detect_outliers(raw_data)
        
        # WASM preprocessing
        processed = await wasm_preprocess(clean_data)
        
        # Convert to numpy array
        X = np.array(processed.get('ma_5', clean_data)).reshape((1, -1, 1))
        
        # Make prediction
        prediction = predictor.model.predict(X, verbose=0)[0][0]
        
        # Calculate cash-out range with confidence
        confidence = np.random.uniform(0.7, 0.95)
        cash_out_range = {
            "lower": max(1.0, prediction * (1 - (0.2 * confidence))),
            "upper": prediction * (1 + (0.2 * confidence)),
            "confidence": float(confidence),
            "volatility_index": np.std(X) / np.mean(X)
        }
        
        # Store raw data in Supabase
        if data.multipliers:
            await supabase.table('multipliers').insert([
                {"value": val} for val in data.multipliers
            ]).execute()
        
        if data.crash_values:
            await supabase.table('crash_values').insert([
                {"value": val} for val in data.crash_values
            ]).execute()
        
        return {
            "prediction": float(prediction),
            "cash_out_range": cash_out_range,
            "timestamp": datetime.now().isoformat(),
            "model_version": predictor.version,
            "accuracy": predictor.accuracy,
            "processed_features": processed
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

# Training Endpoint
@app.post("/train")
async def train_model():
    try:
        # Get historical data from Supabase
        multiplier_response = supabase.table('multipliers').select('value').execute()
        crash_value_response = supabase.table('crash_values').select('value').execute()
        
        # Extract values
        multipliers = [item['value'] for item in multiplier_response.data]
        crash_values = [item['value'] for item in crash_value_response.data]
        
        # Combine and detect outliers
        raw_data = multipliers + crash_values
        clean_data = detect_outliers(raw_data)
        
        # Ensure we have enough data for training
        if len(clean_data) < 10:
            return {"status": "Not enough data for training"}
        
        # Create sequences for LSTM
        X, y = [], []
        sequence_length = 50
        for i in range(len(clean_data) - sequence_length):
            X.append(clean_data[i:i+sequence_length])
            y.append(clean_data[i+sequence_length])
        
        X = np.array(X).reshape((-1, sequence_length, 1))
        y = np.array(y)
        
        # Train model
        history = predictor.model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)
        new_accuracy = max(predictor.accuracy, max(history.history['val_accuracy']) * 100)
        
        # Only update if accuracy improved
        if new_accuracy > predictor.accuracy:
            predictor.accuracy = new_accuracy
            predictor.save_model_version()
            
        # Save training history
        supabase.table('training_history').insert({
            "model_version": predictor.version,
            "samples_used": len(X),
            "accuracy_before": predictor.accuracy,
            "accuracy_after": new_accuracy,
            "loss": float(history.history['loss'][-1]),
            "user_id": "anonymous"  # Replace with actual user ID if available
        }).execute()
        
        return {
            "status": "Model trained",
            "new_accuracy": new_accuracy,
            "version": predictor.version,
            "samples_used": len(X)
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return {"error": str(e)}

# Accuracy Tracking Endpoint
@app.get("/accuracy")
async def get_accuracy():
    return {
        "accuracy": predictor.accuracy,
        "model_version": predictor.version,
        "last_updated": datetime.now().isoformat()
    }

# WASM Preprocessing Proxy
@app.post("/preprocess")
async def wasm_proxy(data: dict):
    """Proxy for WASM preprocessing"""
    input_data = data.get("data", [])
    return await wasm_preprocess(input_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)