"""
Loan Predictor - CORRECTED for One-Hot Encoding
Removed derived features (loan_category, dti_risk)
"""

#from .base_predictor import BasePredictor
from .base_predictor_corrected import BasePredictor
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any


class LoanPredictor(BasePredictor):
    
    def preprocess(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess loan application data - CORRECTED VERSION
        
        ONLY uses base features (no derived features):
        - loan_amnt
        - dti
        - annual_inc
        - addr_state
        - emp_length
        """
        
        # Extract ONLY base features
        loan_amnt = input_data['loan_amnt']
        dti = input_data['dti']
        annual_inc = input_data.get('annual_inc', 50000)
        addr_state = input_data['addr_state']
        emp_length = input_data.get('emp_length', 'Unknown')
        
        # Create features dictionary with ONLY base features
        features = {
            'loan_amnt': loan_amnt,
            'dti': dti,
            'annual_inc': annual_inc,
            'addr_state': addr_state,
            'emp_length': emp_length
        }
        
        # Return as DataFrame (will be one-hot encoded in base class)
        return pd.DataFrame([features])
    
    
    def postprocess(self, prediction: int, confidence: float, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add business logic and enrich prediction"""
        
        recommendation = "APPROVED" if prediction == 1 else "REJECTED"
        display_confidence = confidence if prediction == 1 else (1 - confidence)
        
        estimated_grade = self._estimate_grade(confidence, input_data['dti'], input_data['loan_amnt']) if prediction == 1 else None
        estimated_rate = self._estimate_rate(estimated_grade) if estimated_grade else None
        risk_level = self._assess_risk(input_data['dti'], input_data['loan_amnt'], confidence)
        
        return {
            "recommendation": recommendation,
            "confidence": round(display_confidence, 4),
            "confidence_percentage": f"{display_confidence * 100:.2f}%",
            "loan_amount": input_data['loan_amnt'],
            "risk_assessment": risk_level,
            "estimated_grade": estimated_grade,
            "estimated_rate": estimated_rate,
            "details": {
                # Keep these for display only (not used in model)
                "loan_category": self._categorize_loan_amount(input_data['loan_amnt']),
                "dti_risk": self._categorize_dti(input_data['dti']),
                "state": input_data['addr_state'],
                "model_confidence": round(confidence, 4)
            }
        }
    
    @staticmethod
    def _categorize_loan_amount(amount: float) -> str:
        """Categorize loan amount - FOR DISPLAY ONLY"""
        if amount <= 5000:
            return 'small'
        elif amount <= 15000:
            return 'medium'
        elif amount <= 25000:
            return 'large'
        else:
            return 'jumbo'
    
    @staticmethod
    def _categorize_dti(dti: float) -> str:
        """Categorize DTI - FOR DISPLAY ONLY"""
        if dti <= 15:
            return 'low'
        elif dti <= 28:
            return 'medium'
        elif dti <= 36:
            return 'high'
        else:
            return 'very_high'
    
    @staticmethod
    def _estimate_grade(confidence: float, dti: float, loan_amnt: float) -> str:
        """Estimate loan grade"""
        if confidence >= 0.85 and dti < 20:
            return 'A'
        elif confidence >= 0.75 and dti < 28:
            return 'B'
        elif confidence >= 0.60 and dti < 32:
            return 'C'
        elif confidence >= 0.50:
            return 'D'
        else:
            return 'E-F'
    
    @staticmethod
    def _estimate_rate(grade: str) -> str:
        """Estimate interest rate range"""
        rate_map = {
            'A': '5.3% - 8.0%',
            'B': '8.0% - 11.0%',
            'C': '11.0% - 14.0%',
            'D': '14.0% - 18.0%',
            'E-F': '18.0% - 31.0%'
        }
        return rate_map.get(grade, '10.0% - 20.0%')
    
    @staticmethod
    def _assess_risk(dti: float, loan_amnt: float, confidence: float) -> str:
        """Assess overall risk"""
        if confidence >= 0.80:
            return 'LOW'
        elif confidence >= 0.60:
            return 'MEDIUM'
        else:
            return 'HIGH'
