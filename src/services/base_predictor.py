"""
Base class for all ML predictors - extensible for multiple recommendation engines
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
        self.encoders = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model and preprocessing artifacts"""
        try:
            #self.model = joblib.load(self.model_path / "recommendation_model.pkl")
            self.model = joblib.load(self.model_path / "loan_model.pkl")
            self.encoders = joblib.load(self.model_path / "encoder.pkl")
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            #self.feature_names = joblib.load(self.model_path / "feature_names.pkl")
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
        # 1. Preprocess (engine-specific)
        processed_data = self.preprocess(input_data)
        print("\n=== STEP 1: PREPROCESSED ===")
        print("Shape:", processed_data.shape)
        print("Data:\n", processed_data)
        print("Data types:\n", processed_data.dtypes)
        
        # DEBUG: Print what we're sending to the model
        print("\n=== DEBUG INFO ===")
        print("Input data:", input_data)
        print("Processed data shape:", processed_data.shape)
        print("Processed data columns:", processed_data.columns.tolist())
        print("Processed data values:", processed_data.values[0])
        print("Expected features:", self.feature_names)
        print("==================\n")

        # 2. Encode categorical features
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
        
           scaled_data = pd.DataFrame(
            scaled_array,
            columns=encoded_data.columns
          )
           print("\n=== STEP 3: SCALED (DataFrame) ===")
           print("Shape:", scaled_data.shape)
           print("Data:\n", scaled_data)
        
        except Exception as e:
         print(f"\n!!! SCALING ERROR: {e}")
         raise

        # 4. Make prediction
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        confidence = probabilities[1]
        
        # 5. Postprocess (engine-specific)
        result = self.postprocess(prediction, confidence, input_data)
        
        return result
    
    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """Encode categorical features using stored encoders"""
       encoded = df.copy()
       for col, encoder in self.encoders.items():
           if col in encoded.columns:
              try:
                   # Get the value
                   values = encoded[col].values

                   # Check if values are in encoder's known classes
                   unknown_mask = ~pd.Series(values).isin(encoder.classes_)
                
                   if unknown_mask.any():
                    # Replace unknown values with the most common class
                    print(f"Warning: Unknown values in {col}, using default")
                    values[unknown_mask] = encoder.classes_[0]
                
                   # Encode
                   encoded[col] = encoder.transform(values)
                
              except Exception as e:
               print(f"Error encoding {col}: {e}, using fallback")
               encoded[col] = 0
    
       # Ensure all columns are numeric
       return encoded.astype(float)
    
    def batch_predict(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple predictions"""
        return [self.predict(data) for data in input_list]
