import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.stats import iqr
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
import optuna
import shap
import lime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashTrainer")

class CrashTrainer:
    def __init__(self, predictor=None):
        self.predictor = predictor or CrashPredictor()
        self.data_processor = DataProcessor()
        self.hyperparameter_tuner = HyperparameterTuner()
        self.model_analyzer = ModelAnalyzer()
        self.data_augmenter = DataAugmenter()
        
    def train(self, data, epochs=10, batch_size=32):
        """Train model with advanced features"""
        try:
            # Preprocess data
            processed_data = self.data_processor.preprocess(data)
            
            # Split data for cross-validation
            kf = KFold(n_splits=5)
            
            histories = []
            for train_index, val_index in kf.split(processed_data['X']):
                X_train, X_val = processed_data['X'][train_index], processed_data['X'][val_index]
                y_train, y_val = processed_data['y'][train_index], processed_data['y'][val_index]
                
                # Train model
                history = self.predictor.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=self._get_callbacks(),
                    verbose=0
                )
                histories.append(history)
                
            # Calculate average accuracy across folds
            avg_accuracy = np.mean([history.history['val_accuracy'][-1] for history in histories]) * 100
            
            # Update model if accuracy improves
            if avg_accuracy > self.predictor.accuracy:
                self.predictor.accuracy = avg_accuracy
                self.predictor.update_model(processed_data)
                
            return {
                "status": "Training complete",
                "accuracy": avg_accuracy,
                "histories": histories
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"error": str(e)}

    def auto_train(self, data):
        """Automated training with hyperparameter tuning"""
        try:
            # Optimize hyperparameters
            study = self.hyperparameter_tuner.optimize(data)
            
            # Train with best parameters
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
        """Get training callbacks"""
        return [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=1)
        ]


class DataProcessor:
    def preprocess(self, data):
        """Comprehensive data preprocessing"""
        # Convert to numpy arrays
        X = np.array(data['multipliers'] + data['crash_values'])
        X = X.reshape((1, -1, 1))
        
        # Detect and remove outliers
        X = self._remove_outliers(X)
        
        # Normalize data
        X = self._normalize(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X)
        
        return {"X": X_seq, "y": y_seq}

    def _remove_outliers(self, data):
        """Remove extreme outliers using IQR method"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def _normalize(self, data):
        """Normalize data between 0 and 1"""
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _create_sequences(self, data, seq_length=50):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)


class HyperparameterTuner:
    def optimize(self, data, n_trials=20):
        """Optimize hyperparameters using Optuna"""
        try:
            study = optuna.create_study(direction='maximize')
            objective = self._create_objective(data)
            study.optimize(objective, n_trials=n_trials)
            return study
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return None

    def _create_objective(self, data):
        """Create Optuna objective function"""
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': trial.suggest_int('epochs', 5, 20)
            }
            
            # Train model with trial parameters
            model = self._build_model(params)
            history = model.fit(
                data['X'], data['y'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_split=0.2,
                verbose=0
            )
            
            # Return validation accuracy
            return max(history.history['val_accuracy'])
            
        return objective

    def _build_model(self, params):
        """Build model for hyperparameter tuning"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(None, 1)),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['accuracy']
        )
        
        return model


class ModelAnalyzer:
    def explain_prediction(self, model, data):
        """Explain individual predictions using SHAP and LIME"""
        try:
            # SHAP explanation
            explainer = shap.DeepExplainer(model)
            shap_values = explainer.shap_values(data)
            
            # LIME explanation
            lime_explainer = lime.LimeTabularExplainer(
                data,
                mode="regression"
            )
            lime_explanation = lime_explainer.explain_instance(
                data[0],
                model.predict
            )
            
            return {
                "shap": shap_values,
                "lime": lime_explanation.as_list()
            }
            
        except Exception as e:
            logger.error(f"Model explanation failed: {str(e)}")
            return {"error": str(e)}


class DataAugmenter:
    def augment(self, data):
        """Generate synthetic data for training"""
        augmented_data = {
            "X": [],
            "y": []
        }
        
        # Add noise
        augmented_data["X"].extend(data["X"] * (1 + np.random.normal(0, 0.01, data["X"].shape)))
        
        # Time shift
        augmented_data["X"].extend(np.roll(data["X"], shift=1))
        
        return augmented_data