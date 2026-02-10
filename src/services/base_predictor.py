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
            self.model = joblib.load(self.model_path / "recommendation_model.pkl")
            self.encoders = joblib.load(self.model_path / "encoders.pkl")
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            self.feature_names = joblib.load(self.model_path / "feature_names.pkl")
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
        
        # 2. Encode categorical features
        encoded_data = self._encode_features(processed_data)
        
        # 3. Scale features
        scaled_data = self.scaler.transform(encoded_data)
        
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
                value = encoded[col].iloc[0]
                
                # Special handling for zip_code - use mode (most common value)
                if col == 'zip_code' and value not in encoder.classes_:
                    # Use the most common zip code from training data
                    encoded[col] = encoder.transform([encoder.classes_[0]])[0]
                    print(f"Warning: Unknown zip_code '{value}', using default")
                # For other categorical features, try 'Unknown' fallback
                elif value not in encoder.classes_:
                    if 'Unknown' in encoder.classes_:
                        encoded[col] = encoder.transform(['Unknown'])[0]
                    else:
                        # Use the first class as default
                        encoded[col] = encoder.transform([encoder.classes_[0]])[0]
                        print(f"Warning: Unknown value for {col}, using default")
                else:
                    # Normal encoding
                    encoded[col] = encoder.transform(encoded[col])
                    
            except Exception as e:
                print(f"Error encoding {col}: {e}, using fallback")
                encoded[col] = 0
    
      return encoded
    
    def batch_predict(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple predictions"""
        return [self.predict(data) for data in input_list]
