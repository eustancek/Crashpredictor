# training.py
# --- Fetch HF Token from Colab Secret at the very beginning ---
import os
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('Hugging')
    print("HF_TOKEN fetched from Colab secret 'Hugging' and set as environment variable.")
except Exception as e:
    print(f"Could not fetch HF_TOKEN from Colab secret: {e}. Hugging Face uploads might fail if token is not set elsewhere.")
# --- End HF Token Fetch ---
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.stats import zscore, iqr
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # Import Input
import optuna
import shap
import lime
import logging
# --- Updated database imports ---
from sqlalchemy import create_engine, text
import psycopg2
# --- End updated database imports ---
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime
# --- Import Gradio ---
import gradio as gr
# --- Import Hugging Face Hub ---
from huggingface_hub import HfApi, hf_hub_download, create_repo
# --- 1. Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashTrainer")
# --- 2. Supabase Configuration ---
# Hardcoded Supabase URL and Pooler details
SUPABASE_PROJECT_REF = "fawcuwcqfwzvdoalcocx"
SUPABASE_DB_PASSWORD = "DMMWovTqFUm5RAfY" # <-- Your correct database password
SUPABASE_DB_USER = f"postgres.{SUPABASE_PROJECT_REF}"
SUPABASE_DB_HOST = "aws-0-ca-central-1.pooler.supabase.com" # Pooler host
SUPABASE_DB_PORT = "5432" # Pooler port
SUPABASE_DB_NAME = "postgres"
# --- 3. CORRECTED PostgreSQL connection string for Pooler Connection ---
# Explicitly build the connection string with verified credentials
DATABASE_URL = (
    f"postgresql://{SUPABASE_DB_USER}:{SUPABASE_DB_PASSWORD}@"
    f"{SUPABASE_DB_HOST}:{SUPABASE_DB_PORT}/{SUPABASE_DB_NAME}"
    f"?sslmode=require"
)
logger.info(f"Constructed DATABASE_URL (user part): postgresql://{SUPABASE_DB_USER}:****@{SUPABASE_DB_HOST}:{SUPABASE_DB_PORT}/{SUPABASE_DB_NAME}?sslmode=require")
# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
# --- End of Database Connection Setup ---
# --- 4. Classes (CrashTrainer, DataProcessor, etc.) ---
# Ensure the KFold n_splits is reduced (e.g., to 3) as seen in your logs.
# Ensure mock data generation logic is present in DataProcessor.load_and_clean_data.
# Ensure model uses Input layer correctly.
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
                    raise ValueError("No data available for training (even after mock data fallback).")
            processed_data = self.data_processor.preprocess(data)
            if processed_data["X"].shape[0] < 50: # Check for sufficient samples
                raise ValueError("Insufficient samples for training after preprocessing.")
            # Reduced KFold splits to handle potentially small datasets
            kf = KFold(n_splits=3)
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
                # Fit the model
                history = self.predictor.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=self._get_callbacks(),
                    verbose=0 # Reduce training output verbosity
                )
                histories.append(history)
            # Calculate average accuracy
            avg_accuracy = np.mean([h.history['val_accuracy'][-1] for h in histories]) * 100
            # Save model if it's better or if incremental training is specified
            if incremental or avg_accuracy > self.predictor.accuracy:
                self.predictor.accuracy = avg_accuracy
                self.predictor.save_model() # Save the model (now includes upload)
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
            # Perform hyperparameter optimization
            study = self.hyperparameter_tuner.optimize(data)
            best_params = study.best_params
            # Train using the best parameters found
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
        """Define Keras callbacks"""
        return [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=1)
        ]
class DataProcessor:
    # --- START OF UPDATED load_and_clean_data METHOD ---
    def load_and_clean_data(self):
        """Load data from Supabase and clean it"""
        try:
            # --- 1. Query to fetch data ---
            query = """
            SELECT crash_value, created_at
            FROM crash_data
            ORDER BY created_at ASC
            LIMIT 200
            """
            logger.info("Attempting to load data from Supabase database...")
            # --- Explicitly test the connection using engine.connect() context manager ---
            # This mimics the successful direct connection test more closely.
            logger.info("Performing diagnostic connection test...")
            with engine.connect() as connection:
                # Test with a very simple query first
                test_result = connection.execute(text("SELECT 1"))
                test_value = test_result.fetchone()
                logger.info(f"Diagnostic connection test successful. SELECT 1 returned: {test_value[0] if test_value else 'None'}")
                # Optional: Test with a version query (like in the previous version)
                # result = connection.execute(text("SELECT version();"))
                # db_version = result.fetchone()
                # logger.info(f"Database connection successful. PostgreSQL version: {db_version[0] if db_version else 'Unknown'}")
            # --- 2. Attempt to load data from the database ---
            # If the above test passes, this *should* work.
            df = pd.read_sql(query, con=engine)
            logger.info(f"Database query executed. Rows returned: {len(df)}")
            # --- 3. Check if data was loaded ---
            if df.empty:
                logger.warning("Database query returned 0 rows. Attempting mock data generation.")
                # --- 4a. Mock Data Fallback (DB returned empty result) ---
                mock_data = self._generate_mock_data()
                logger.info("Using mock data generated from empty query result.")
                # Process the mock data before returning
                mock_data = self._remove_outliers(mock_data)
                mock_data['scaled'] = self._normalize(mock_data['crash_value'])
                logger.info("Mock data processing complete.")
                return mock_data # Return processed mock data
            else:
                logger.info("Data successfully loaded from database.")
                # --- 5. Process real data ---
                # Outlier removal and normalization
                df = self._remove_outliers(df)
                df['scaled'] = self._normalize(df['crash_value'])
                logger.info(f"Real data processing complete. Final shape: {df.shape}")
                return df # Return the cleaned real data
        except Exception as db_error:
            # --- 4b. Mock Data Fallback (DB connection/query failed) ---
            logger.error(f"Database connection or query failed: {db_error}")
            logger.warning("Database access failed. Generating mock data as fallback.")
            mock_data = self._generate_mock_data()
            # Apply basic processing to mock data
            try:
                mock_data = self._remove_outliers(mock_data)
                mock_data['scaled'] = self._normalize(mock_data['crash_value'])
                logger.info("Mock data processing complete after DB error.")
            except Exception as proc_error:
                logger.warning(f"Could not process mock data after DB error, returning raw mock: {proc_error}")
            logger.info("Using mock data generated from database error.")
            return mock_data # Return mock data if DB fails
    # --- END OF UPDATED load_and_clean_data METHOD ---
    # --- START OF NEW _generate_mock_data HELPER METHOD ---
    def _generate_mock_data(self):
        """Helper to create consistent mock data."""
        logger.info("Generating 200 rows of mock crash data...")
        mock_data = pd.DataFrame({
            "crash_value": np.random.randn(200),
            "created_at": pd.date_range(start="2023-01-01", periods=200, freq="D")
        })
        logger.info(f"Mock data generated with shape: {mock_data.shape}")
        return mock_data
    # --- END OF NEW _generate_mock_data HELPER METHOD ---
    def _remove_outliers(self, df, z_threshold=3, iqr_multiplier=1.5):
        """Remove extreme outliers using Z-score and IQR"""
        # Z-score filtering
        # Calculate the absolute Z-scores for crash_value
        z_scores = np.abs(zscore(df['crash_value']))
        # Filter the DataFrame to keep rows with Z-scores below the threshold
        df = df[z_scores < z_threshold]
        # IQR filtering
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df['crash_value'].quantile(0.25)
        Q3 = df['crash_value'].quantile(0.75)
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        # Filter the DataFrame to remove outliers based on IQR
        df = df[~((df['crash_value'] < (Q1 - iqr_multiplier * IQR)) |
                 (df['crash_value'] > (Q3 + iqr_multiplier * IQR)))]
        return df # Return the DataFrame without outliers
    def _normalize(self, data):
        """Normalize data between 0 and 1"""
        # Prevent division by zero if max == min
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min == 0:
             return np.zeros_like(data) # Return zeros if all values are the same
        # Normalize the data
        return (data - data_min) / (data_max - data_min + 1e-7) # Add epsilon to avoid division by zero
    def preprocess(self, df):
        """Create sequences for LSTM training"""
        try:
            scaled_values = df['scaled'].values.reshape(-1, 1)
            seq_length = 50 # Length of each sequence
            X, y = [], [] # Initialize lists for sequences and targets
            # Loop through the scaled values to create sequences
            for i in range(len(scaled_values) - seq_length):
                X.append(scaled_values[i:i+seq_length]) # Append sequence
                y.append(scaled_values[i+seq_length]) # Append target value
            X = np.array(X)
            y = np.array(y)
            # Check if there are sufficient samples after preprocessing
            if X.shape[0] < 1: # Adjusted check
                raise ValueError("Insufficient samples after preprocessing")
            # Return a dictionary with sequences, targets, and raw values
            return {"X": X, "y": y, "raw": df['crash_value'].values}
        except Exception as e:
            # Log any errors that occur during preprocessing
            logger.error(f"Preprocessing failed: {str(e)}")
            # Return empty arrays in case of failure
            return {"X": np.array([]), "y": np.array([])}
class CrashPredictor:
    def __init__(self):
        # Initialize accuracy to 0
        self.accuracy = 0
        # Get the latest model version
        self.version = self._get_latest_model_version()
        # Load or build the model
        self.model = self._load_or_build_model()
    # --- FIXED Model Version Detection ---
    def _get_latest_model_version(self):
        """Get the highest version number of saved models"""
        try:
            os.makedirs("models", exist_ok=True) # Ensure directory exists
            model_files = os.listdir('models')
            versions = []
            for f in model_files:
                if f.startswith('crash_model_') and f.endswith('.keras'):
                    try:
                        version_str = f.split('_')[2].split('.')[0]
                        version_num = int(version_str)
                        versions.append(version_num)
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Skipping file '{f}' during version detection due to unexpected name format: {e}")
            return max(versions) if versions else 0
        except Exception as e:
            logger.warning(f"Model version detection failed unexpectedly: {str(e)}")
            return 0
    # --- END FIXED Model Version Detection ---
    def _load_or_build_model(self):
        """Load an existing model or build a new one"""
        model_path = f"models/crash_model_{self.version}.keras"
        if os.path.exists(model_path):
            try:
                # Load the existing model
                logger.info(f"Loading existing model: {model_path}")
                return load_model(model_path)
            except Exception as e:
                # Log a warning if loading fails
                logger.warning(f"Failed to load model {model_path}: {str(e)}. Building a new one.")
        # --- Fix: Use Input layer to avoid Keras warning ---
        logger.info("Building a new model.")
        # Define the input layer with the correct shape (None for variable timesteps)
        input_layer = Input(shape=(None, 1)) # Correct way to define input for Sequential model
        # Build the Sequential model using the input layer
        model = Sequential([
            input_layer, # Use the defined input layer
            LSTM(128, return_sequences=True), # First LSTM layer
            Dropout(0.3), # Dropout layer for regularization
            LSTM(64), # Second LSTM layer
            Dense(32, activation='relu'), # Dense layer with ReLU activation
            Dense(1) # Output layer for regression
        ])
        # Compile the model with optimizer, loss, and metrics
        model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
        return model # Return the built model
    def save_model(self):
        """Save the current model to a file (.keras format) and manage uploads/cleanup"""
        self.version += 1
        model_path = f"models/crash_model_{self.version}.keras"
        try:
            # Save the model in the native Keras format
            self.model.save(model_path)
            # Log successful saving
            logger.info(f"Model saved successfully as version {self.version} to {model_path}")
            # --- NEW: Call upload and cleanup function ---
            upload_latest_model_to_hf()
            return True # Indicate success
        except Exception as e:
            # Log an error if saving fails
            logger.error(f"Model save failed: {str(e)}")
            return False # Indicate failure
class HyperparameterTuner:
    def optimize(self, data=None, n_trials=20):
        """Perform hyperparameter optimization using Optuna"""
        try:
            study = optuna.create_study(direction='maximize')
            objective = self._create_objective(data)
            study.optimize(objective, n_trials=n_trials)
            return study # Return the optimized study
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return None # Return None in case of failure
    def _create_objective(self, data):
        """Create the objective function for Optuna"""
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
        return objective # Return the objective function
class ModelAnalyzer:
    def explain_prediction(self, model, data):
        """Explain a model's prediction using SHAP and LIME"""
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
        augmented_X = []
        augmented_y = []
        augmented_X.extend((data["X"] * (1 + np.random.normal(0, 0.01, data["X"].shape))).tolist())
        augmented_X.extend(np.roll(data["X"], shift=1).tolist())
        return {"X": np.array(augmented_X), "y": np.array(augmented_y)}
# --- 5. Hugging Face Integration Functions ---
# --- Updated with your specific details ---
HF_REPO_ID = "eustancek/Google-colab" # Your HF repo
HF_MODEL_PATH = "models/latest_model.keras"      # Consistent path on HF for the latest model
MAX_MODELS_TO_KEEP = 3                          # Keep last 3 versions locally & on HF
def get_hf_token():
    """Retrieve the Hugging Face token from environment variables."""
    return os.getenv('HF_TOKEN') # This will now use the token set from the secret
def upload_latest_model_to_hf():
    """Uploads the latest trained model to Hugging Face with a fixed name, keeps the last N versions, and cleans up."""
    token = get_hf_token()
    if not token:
        logger.warning("HF_TOKEN not found. Skipping Hugging Face upload and cleanup.")
        return
    api = HfApi(token=token)
    try:
        # --- 1. Find the latest local model file ---
        os.makedirs("models", exist_ok=True)
        model_files = [f for f in os.listdir('models') if f.startswith('crash_model_') and f.endswith('.keras')]
        if not model_files:
            logger.info("No local model files found to upload.")
            return
        # Sort by version number (assuming version is numeric)
        model_files.sort(key=lambda f: int(f.split('_')[2].split('.')[0]), reverse=True)
        latest_local_model_file = model_files[0]
        latest_local_model_path = os.path.join("models", latest_local_model_file)
        latest_version_num = int(latest_local_model_file.split('_')[2].split('.')[0])
        logger.info(f"Latest local model identified: {latest_local_model_file}")
        # --- 2. Ensure repository exists ---
        repo_type = "space"  # We're working with a space repository
        try:
            # Try to get repo info to check if it exists
            api.repo_info(repo_id=HF_REPO_ID, repo_type=repo_type)
            logger.info(f"Repository {HF_REPO_ID} exists.")
        except Exception as e:
            logger.warning(f"Repository {HF_REPO_ID} does not exist. Creating it now...")
            try:
                api.create_repo(repo_id=HF_REPO_ID, repo_type=repo_type, private=False)
                logger.info(f"Repository {HF_REPO_ID} created successfully.")
            except Exception as create_error:
                logger.error(f"Failed to create repository: {create_error}")
                return
        # --- 3. Upload the latest model with the fixed HF path ---
        api.upload_file(
            path_or_fileobj=latest_local_model_path,
            path_in_repo=HF_MODEL_PATH,
            repo_id=HF_REPO_ID,
            repo_type=repo_type,
        )
        logger.info(f"Uploaded latest model '{latest_local_model_file}' to Hugging Face as '{HF_MODEL_PATH}'.")
        # --- 4. Delete older local models ---
        if len(model_files) > MAX_MODELS_TO_KEEP:
            models_to_delete_local = model_files[MAX_MODELS_TO_KEEP:]
            for model_file in models_to_delete_local:
                model_path = os.path.join("models", model_file)
                try:
                    os.remove(model_path)
                    logger.info(f"Deleted old local model file: {model_file}")
                except OSError as e:
                    logger.error(f"Error deleting local model file {model_file}: {e}")
        # --- 5. Delete older versions on Hugging Face ---
        # List existing files in the 'models' directory on HF
        try:
            hf_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=repo_type)
            model_files_on_hf = [f for f in hf_files if f.startswith('models/crash_model_') and f.endswith('.keras')]
            # Extract version numbers
            hf_versions = []
            for f in model_files_on_hf:
                 try:
                     version_str = f.split('_')[2].split('.')[0]
                     version_num = int(version_str)
                     hf_versions.append((version_num, f))
                 except (IndexError, ValueError):
                     logger.warning(f"Skipping HF file '{f}' during cleanup due to unexpected name format.")
            # Sort by version number descending
            hf_versions.sort(key=lambda x: x[0], reverse=True)
            # Identify versions to delete
            if len(hf_versions) > MAX_MODELS_TO_KEEP:
                versions_to_delete_hf = hf_versions[MAX_MODELS_TO_KEEP:]
                for version_num, file_path in versions_to_delete_hf:
                    try:
                        api.delete_file(
                            path_in_repo=file_path,
                            repo_id=HF_REPO_ID,
                            repo_type=repo_type,
                            token=token
                        )
                        logger.info(f"Deleted old model version {version_num} from Hugging Face: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting model version {version_num} from Hugging Face ({file_path}): {e}")
            else:
                 logger.info("No old model versions to delete on Hugging Face.")
        except Exception as e:
            logger.error(f"Error listing or deleting old models on Hugging Face: {e}")
    except Exception as e:
        logger.error(f"Error during Hugging Face upload or cleanup process: {e}")
# --- End Hugging Face Integration Functions ---
# --- 6. Main execution block ---
# Ensure directories exist when the script is run directly.
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    logger.info("Ensured 'models' directory exists.")
# --- 7. Gradio Interface Definition ---
# This part is crucial for Hugging Face Spaces
def predict_crash(input_string):
    """Gradio prediction function."""
    # Your prediction logic here (can be simplified for testing)
    try:
        # Simple mock prediction for testing
        input_list = [float(x.strip()) for x in input_string.split(',')]
        if len(input_list) != 50: # Example check
            return f"Error: Expected 50 values, got {len(input_list)}."
        # Mock prediction: average of inputs
        prediction = sum(input_list) / len(input_list)
        return f"Mock Predicted Crash Value: {prediction:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"
# Define Gradio Interface
logger.info("Defining Gradio interface...")
iface = gr.Interface(
    fn=predict_crash,
    inputs=gr.Textbox(lines=5, placeholder="Enter 50 comma-separated crash values...", label="Input Data (Last 50 Values)"),
    outputs=gr.Textbox(label="Prediction"),
    title="Crash Predictor (Demo)",
    description="Enter the last 50 crash values to predict the next one. This is a demo placeholder.",
    examples=[
        ["0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0"]
    ]
)
# --- CRITICAL FOR HUGGING FACE SPACES ---
# Assign the interface object to the variable 'demo'.
# Hugging Face Spaces looks for this specific name to launch the app.
logger.info("Assigning Gradio interface to 'demo' variable...")
demo = iface
logger.info("'demo' variable assigned. Hugging Face should launch the app now.")
# DO NOT call demo.launch() or iface.launch() here.
# Hugging Face handles the launching process.
# --- END CRITICAL PART ---
logger.info("app.py module loaded successfully. Waiting for Hugging Face to launch 'demo'.")
