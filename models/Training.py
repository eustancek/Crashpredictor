import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.stats import zscore, iqr
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
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
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://fawcuwcqfwzvdoalcocx.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhd2N1d2NxZnd6dmRvYWxjb2N4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA4NDY3MjYsImV4cCI6MjA2NjQyMjcyNn0.5NCGUTGpPm7w2Jv0GURMKmGh-EQ7WztNLs9")  # Your actual Anon key (without "MD5_nSjc")

# Fixed PostgreSQL connection string (Transaction Pooler)
engine = create_engine(
    f"postgresql://"
    f"{SUPABASE_KEY.split('.')[0]}:"  # Username is first part of the key
    f"{SUPABASE_KEY}@"  # Full key as password
    f"aws-0-ca-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
)

class CrashTrainer:
    def __init__(self, predictor=None):
        self.predictor = predictor or CrashPredictor()
        self.data_processor = DataProcessor()
        self.hyperparameter_tuner = HyperparameterTuner()
        self.model_analyzer = ModelAnalyzer()
        self.data_augmenter = DataAugmenter()
        self.scaler = MinMaxScaler()

    def train(self, data=None, epochs=10, batch_size=32, incremental=False):
        try:
            if data is None:
                data = self.data_processor.load_and_clean_data()
                if data.empty:
                    raise ValueError("No data available for training")
            processed_data = self.data_processor.preprocess(data)
            if processed_data["X"].shape[0] < 50:
                raise ValueError("Insufficient samples for training")
            kf = KFold(n_splits=3)  # Reduced splits to handle small datasets
            histories = []
            for train_index, val_index in kf.split(processed_data['X']):
                X_train, X_val = (
                    processed_data['X'][train_index],
                    processed_data['X'][val_index]
                )
                y_train, y_val = (
                    processed_data['y'][train_index],
                    processed_data['y'][val_index]
                )
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
                self.predictor.save_model()
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
        try:
            query = """
            SELECT crash_value, created_at 
            FROM crash_data 
            ORDER BY created_at ASC
            LIMIT 200
            """
            df = pd.read_sql(query, con=engine)
            if df.empty:
                logger.warning("No real data found. Generating mock data...")
                mock_data = pd.DataFrame({
                    "crash_value": np.random.randn(200),
                    "created_at": pd.date_range(start="2023-01-01", periods=200, freq="D")
                })
                df = mock_data
            df = self._remove_outliers(df)
            df['scaled'] = self._normalize(df['crash_value'])
            return df
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    def _remove_outliers(self, df, z_threshold=3, iqr_multiplier=1.5):
        df = df[(np.abs(zscore(df['crash_value'])) < z_threshold)]
        Q1 = df['crash_value'].quantile(0.25)
        Q3 = df['crash_value'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['crash_value'] < (Q1 - iqr_multiplier * IQR)) | 
                 (df['crash_value'] > (Q3 + iqr_multiplier * IQR)))]
        return df

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min() + 1e-7)

    def preprocess(self, df):
        try:
            scaled_values = df['scaled'].values.reshape(-1, 1)
            seq_length = 50
            X, y = [], []
            for i in range(len(scaled_values) - seq_length):
                X.append(scaled_values[i:i+seq_length])
                y.append(scaled_values[i+seq_length])
            X = np.array(X)
            y = np.array(y)
            if X.shape[0] < 1:
                raise ValueError("Insufficient samples after preprocessing")
            return {"X": X, "y": y, "raw": df['crash_value'].values}
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return {"X": np.array([]), "y": np.array([])}

class CrashPredictor:
    def __init__(self):
        self.accuracy = 0
        self.version = self._get_latest_model_version()
        self.model = self._load_or_build_model()

    def _get_latest_model_version(self):
        try:
            os.makedirs("models", exist_ok=True)  # Auto-create directory
            versions = [int(f.split('_')[1].split('.')[0]) 
                        for f in os.listdir('models') 
                        if f.startswith('crash_model_')]
            return max(versions) if versions else 0
        except Exception as e:
            logger.warning(f"Model version detection failed: {str(e)}")
            return 0

    def _load_or_build_model(self):
        model_path = f"models/crash_model_{self.version}.h5"
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}")

        input_layer = Input(shape=(None, 1))
        model = Sequential([
            input_layer,
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
        return model

    def save_model(self):
        self.version += 1
        model_path = f"models/crash_model_{self.version}.h5"
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
            return result.get('accuracy', 0)
        return objective

class ModelAnalyzer:
    def explain_prediction(self, model, data):
        try:
            explainer = shap.DeepExplainer(model)
            shap_values = explainer.shap_values(data)
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(data),
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
        X_aug = []
        y_aug = []
        X_aug.extend((data["X"] * (1 + np.random.normal(0, 0.01, data["X"].shape))).tolist())
        X_aug.extend(np.roll(data["X"], shift=1).tolist())
        return {"X": np.array(X_aug), "y": np.array(y_aug)}

# Ensure models directory exists on startup
def ensure_directories():
    os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
