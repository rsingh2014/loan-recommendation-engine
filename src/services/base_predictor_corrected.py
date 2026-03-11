"""
Base class for all ML predictors - CORRECTED for One-Hot Encoding
"""

from abc import ABC, abstractmethod
from pathlib import Path
import joblib
from typing import Dict, Any, List
import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for ML prediction services"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.encoder = None  # Changed from encoders to encoder (single OneHotEncoder)
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model and preprocessing artifacts"""
        try:
            self.model = joblib.load(self.model_path / "loan_model.pkl")
            self.encoder = joblib.load(self.model_path / "encoder.pkl")  # OneHotEncoder
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            self.feature_names = joblib.load(self.model_path / "features.pkl")
            self.metadata = joblib.load(self.model_path / "model_metadata.pkl")
            print(f"✓ Model loaded: {self.metadata['model_name']}")
        except FileNotFoundError as e:
            raise RuntimeError(f"Model artifacts not found: {e}")
    
    @abstractmethod
    def preprocess(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def postprocess(self, prediction: int, confidence: float, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add business logic and enrich prediction - must be implemented by subclasses"""
        pass
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main prediction pipeline - reusable across all engines"""
        
        # 1. Preprocess (engine-specific) - returns DataFrame with base features
        processed_data = self.preprocess(input_data)
        print("\n=== STEP 1: PREPROCESSED ===")
        print("Shape:", processed_data.shape)
        print("Data:\n", processed_data)
        print("Data types:\n", processed_data.dtypes)
        
        # 2. One-hot encode categorical features
        encoded_data = self._encode_features(processed_data)
        print("\n=== STEP 2: ENCODED ===")
        print("Shape:", encoded_data.shape)
        print("Data:\n", encoded_data)
        print("Data types:\n", encoded_data.dtypes)
        print("Any NaN?", encoded_data.isna().any())
        
        # 3. Scale features
        try:
            scaled_array = self.scaler.transform(encoded_data)
            print("\n=== STEP 3: SCALED (array) ===")
            print("Shape:", scaled_array.shape)
            print("Data:", scaled_array)
            
            # Convert back to DataFrame for clarity
            scaled_data = pd.DataFrame(
                scaled_array,
                columns=encoded_data.columns,
                index=encoded_data.index
            )
            print("\n=== STEP 3: SCALED (DataFrame) ===")
            print("Shape:", scaled_data.shape)
            print("Data:\n", scaled_data)
            
        except Exception as e:
            print(f"❌ Scaling failed: {e}")
            print(f"Encoded data shape: {encoded_data.shape}")
            print(f"Encoded columns: {encoded_data.columns.tolist()}")
            print(f"Expected by scaler: {len(self.feature_names)} features")
            raise
        
        # 4. Make prediction
        try:
            prediction = self.model.predict(scaled_array)[0]
            probabilities = self.model.predict_proba(scaled_array)[0]
            confidence = float(max(probabilities))
            
            print("\n=== STEP 4: PREDICTION ===")
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
            print(f"Confidence: {confidence}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise
        
        # 5. Postprocess (engine-specific)
        result = self.postprocess(prediction, confidence, input_data)
        
        return result
    
    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features using saved encoder
        
        Process:
        1. Identify categorical columns (object dtype)
        2. One-hot encode them using saved encoder
        3. Combine with numeric columns
        4. Return in correct feature order
        """
        
        # Separate categorical and numeric columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\n=== ONE-HOT ENCODING ===")
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numeric columns: {numeric_cols}")
        
        if categorical_cols and self.encoder is not None:
            # One-hot encode categorical features
            encoded_array = self.encoder.transform(df[categorical_cols])
            
            # Get feature names from encoder
            feature_names = self.encoder.get_feature_names_out(categorical_cols)
            
            # Create DataFrame for encoded features
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=feature_names,
                index=df.index
            )
            
            print(f"Encoded shape: {encoded_df.shape}")
            print(f"Encoded columns: {list(encoded_df.columns)[:10]}...")
            
            # Combine numeric and encoded categorical
            numeric_df = df[numeric_cols]
            combined = pd.concat([numeric_df, encoded_df], axis=1)
            
            print(f"Combined shape: {combined.shape}")
            print(f"Combined columns: {list(combined.columns)[:10]}...")
            
        else:
            # No categorical columns to encode
            combined = df
            print("No categorical encoding needed")
        
        # Ensure columns match expected feature names (order matters!)
        if list(combined.columns) != list(self.feature_names):
            print(f"\n⚠️  Column order mismatch - reordering to match training")
            print(f"Current order: {list(combined.columns)[:10]}...")
            print(f"Expected order: {list(self.feature_names)[:10]}...")
            
            # Reorder to match training
            combined = combined[self.feature_names]
        
        return combined
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.metadata.get('model_name', 'Unknown'),
            "model_version": "2.0",  # Updated version with one-hot encoding
            "features": self.feature_names,
            "num_features": len(self.feature_names),
            "encoding": "OneHotEncoder",
            "scaler": "StandardScaler",
            "artifact_checks": {
                "model_loaded": self.model is not None,
                "encoder_loaded": self.encoder is not None,
                "scaler_loaded": self.scaler is not None,
                "feature_names_loaded": self.feature_names is not None
            }
        }
