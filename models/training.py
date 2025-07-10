import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.stats import zscore, iqr
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
import optuna
import shap
import lime
import logging
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashTrainer")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://fawcuwcqfwzvdoalcocx.supabase.co ")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")

# Database Connection
# Fixed: Use proper Supabase PostgreSQL connection string
engine = create_engine(f"postgresql://postgres:{SUPABASE_KEY.split('.')[0]}@db.fawcuwcqfwzvdoalcocx.supabase.co:5432/postgres")

class CrashTrainer:
    def __init__(self, predictor=None):
        self.predictor = predictor or CrashPredictor()
        self.data_processor = DataProcessor()
        self.hyperparameter_tuner = HyperparameterTuner()
        self.model_analyzer = ModelAnalyzer()
        self.data_augmenter = DataAugmenter()
        self.scaler = MinMaxScaler()
        
    def train(self, data=None, epochs=10, batch_size=32, incremental=False):
        """Train model with advanced features"""
        try:
            if data is None:
                data = self.data_processor.load_and_clean_data()
            
            processed_data = self.data_processor.preprocess(data)
            
            # Cross-validation
            kf = KFold(n_splits=5)
            histories = []
            
            for train_index, val_index in kf.split(processed_data['X']):
                X_train, X_val = processed_data['X'][train_index], processed_data['X'][val_index]
                y_train, y_val = processed_data['y'][train_index], processed_data['y'][val_index]
                
                history = self.predictor.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=self._get_callbacks(),
                    verbose=0
                )
                histories.append(history)
            
            avg_accuracy = np.mean([h.history['val_accuracy'][-1] for h in histories]) * 100
            
            if incremental or avg_accuracy > self.predictor.accuracy:
                self.predictor.accuracy = avg_accuracy
                self.predictor.save_model()  # Save updated model
                
            return {
                "status": "Training complete",
                "accuracy": avg_accuracy,
                "histories": histories,
                "model_version": self.predictor.version
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"error": str(e)}

    def auto_train(self, data=None):
        """Automated training with hyperparameter tuning"""
        try:
            study = self.hyperparameter_tuner.optimize(data)
            best_params = study.best_params
            
            result = self.train(
                data,
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Auto-training failed: {str(e)}")
            return {"error": str(e)}

    def _get_callbacks(self):
        return [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=1)
        ]


class DataProcessor:
    def load_and_clean_data(self):
        """Load from Supabase and clean data"""
        try:
            query = """
            SELECT crash_value, created_at 
            FROM crash_data 
            WHERE created_at >= NOW() - INTERVAL '30 days'
            ORDER BY created_at ASC
            """
            
            df = pd.read_sql(query, con=engine)
            
            # Remove outliers using both Z-score and IQR
            df = self._remove_outliers(df)
            
            # Scale data
            df['scaled'] = self._normalize(df['crash_value'])
            
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    def _remove_outliers(self, df, threshold=3):
        """Remove extreme outliers using Z-score and IQR"""
        # Z-score filtering
        z_scores = np.abs(zscore(df['crash_value']))
        df = df[z_scores < threshold]
        
        # IQR filtering
        Q1 = df['crash_value'].quantile(0.25)
        Q3 = df['crash_value'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['crash_value'] < (Q1 - 1.5 * IQR)) | 
                 (df['crash_value'] > (Q3 + 1.5 * IQR)))]
        
        return df

    def _normalize(self, data):
        """Normalize data between 0 and 1"""
        return (data - data.min()) / (data.max() - data.min())

    def preprocess(self, df):
        """Create sequences for LSTM training"""
        try:
            scaled_values = df['scaled'].values.reshape(-1, 1)
            seq_length = 50
            
            X, y = [], []
            for i in range(len(scaled_values) - seq_length):
                X.append(scaled_values[i:i+seq_length])
                y.append(scaled_values[i+seq_length])
                
            return {
                "X": np.array(X),
                "y": np.array(y),
                "raw": df['crash_value'].values
            }
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return {"X": np.array([]), "y": np.array([])}


class CrashPredictor:
    def __init__(self):
        self.accuracy = 0
        self.version = self._get_latest_model_version()
        self.model = self._load_or_build_model()
    
    def _get_latest_model_version(self):
        """Get latest model version from disk"""
        try:
            versions = [int(f.split('_')[1].split('.')[0]) for f in os.listdir('models') if f.startswith('crash_model_')]
            return max(versions) if versions else 0
        except Exception as e:
            logger.warning(f"Model version detection failed: {str(e)}")
            return 0
    
    def _load_or_build_model(self):
        """Load existing model or build new one"""
        model_path = f'models/crash_model_{self.version}.h5'
        
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}")
        
        # Build new model if no saved version exists
        return Sequential([
            LSTM(128, return_sequences=True, input_shape=(None, 1)),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    
    def save_model(self):
        """Save model with versioning"""
        self.version += 1
        model_path = f'models/crash_model_{self.version}.h5'
        
        try:
            self.model.save(model_path)
            logger.info(f"Model saved as version {self.version}")
            return True
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
            return False


class HyperparameterTuner:
    def optimize(self, data=None, n_trials=20):
        try:
            study = optuna.create_study(direction='maximize')
            objective = self._create_objective(data)
            study.optimize(objective, n_trials=n_trials)
            return study
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return None

    def _create_objective(self, data):
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': trial.suggest_int('epochs', 5, 20)
            }
            
            trainer = CrashTrainer()
            result = trainer.train(
                data,
                epochs=params['epochs'],
                batch_size=params['batch_size']
            )
            
            # Fixed: Return accuracy directly instead of from history
            return result.get('accuracy', 0)
            
        return objective


class ModelAnalyzer:
    def explain_prediction(self, model, data):
        try:
            explainer = shap.DeepExplainer(model)
            shap_values = explainer.shap_values(data)
            
            lime_explainer = lime.LimeTabularExplainer(
                data,
                mode="regression"
            )
            lime_exp = lime_explainer.explain_instance(
                data[0],
                model.predict
            )
            
            return {
                "shap": shap_values,
                "lime": lime_exp.as_list()
            }
            
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
            return {"error": str(e)}


class DataAugmenter:
    def augment(self, data):
        augmented = {
            "X": [],
            "y": []
        }
        
        # Fixed: Convert numpy arrays to lists before extending
        augmented["X"].extend((data["X"] * (1 + np.random.normal(0, 0.01, data["X"].shape))).tolist())
        augmented["X"].extend(np.roll(data["X"], shift=1).tolist())
        
        return augmented