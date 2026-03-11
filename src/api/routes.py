import logging
"""
API routes - separated from business logic
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.loan_predictor_corrected import LoanPredictor
from services.loan_validator import LoanValidator  # ← ADD THIS

router = APIRouter()

# Initialize predictor (singleton)
loan_predictor = None

def get_loan_predictor():
    global loan_predictor
    if loan_predictor is None:
        loan_predictor = LoanPredictor(model_path="models")
    return loan_predictor

class LoanApplicationRequest(BaseModel):
    loan_amnt: float = Field(..., ge=500, le=50000, description="Loan amount in dollars")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (%)")
    annual_inc: float = Field(..., ge=10000, le=500000, description="Annual income in dollars")
    addr_state: str = Field(..., min_length=2, max_length=2, description="State code")
    emp_length: Optional[str] = None
    
    @validator('addr_state')
    def uppercase_state(cls, v):
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "loan_amnt": 15000,
                "dti": 18.5,
                "annual_inc": 75000,
                "addr_state": "TX",
                "emp_length": "5 years"
            }
        }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/predict")
async def predict_loan(request: LoanApplicationRequest):
    """Get loan recommendation with validation"""
    try:
        predictor = get_loan_predictor()
        result = predictor.predict(request.dict())
        
        # ============================================================
        # APPLY BUSINESS LOGIC VALIDATION
        # ============================================================
        validation_result = LoanValidator.validate_loan(
            loan_amnt=request.loan_amnt,
            annual_inc=request.annual_inc,
            dti=request.dti
        )
        
        # Override model recommendation if validation fails
        final_recommendation = LoanValidator.apply_override_logic(
            model_recommendation=result['recommendation'],
            validation_result=validation_result
        )
        
        # Merge results
        result.update({
            "recommendation": final_recommendation,
            "risk_assessment": validation_result["risk_level"],
            "loan_to_income_ratio": validation_result["loan_to_income_ratio"],
            "risk_flags": validation_result["risk_flags"]
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict")
async def batch_predict(requests: List[LoanApplicationRequest]):
    """Batch prediction endpoint"""
    try:
        predictor = get_loan_predictor()
        results = predictor.batch_predict([r.dict() for r in requests])
        return {"total": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        predictor = get_loan_predictor()
        return {
            "status": "healthy",
            "model_name": predictor.metadata.get('model_name', 'RandomForest'),
            "model_version": predictor.metadata.get('model_version', '2.0'),
            "features": predictor.feature_names,
            "num_features": len(predictor.feature_names),
            "encoding": predictor.metadata.get('encoder', 'OneHotEncoder'),
            "scaler": predictor.metadata.get('scaler', 'StandardScaler'),
            "artifact_checks": {
                "model_loaded": predictor.model is not None,
                "encoder_loaded": predictor.encoder is not None,
                "scaler_loaded": predictor.scaler is not None,
                "feature_names_loaded": predictor.feature_names is not None
            },
            "required_fields": getattr(predictor, 'required_fields', 
                ['loan_amnt', 'dti', 'annual_inc', 'addr_state', 'emp_length'])
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@router.get("/rules")
async def get_validation_rules():
    """Get all validation rules"""
    return {
        "validation_rules": LoanValidator.get_all_rules(),
        "thresholds": {
            "max_loan_to_income": f"{LoanValidator.MAX_LOAN_TO_INCOME}%",
            "max_dti": f"{LoanValidator.MAX_DTI}%",
            "min_annual_income": f"${LoanValidator.MIN_ANNUAL_INCOME:,}",
            "max_loan_ratio": f"{LoanValidator.MAX_LOAN_TO_INCOME_RATIO * 100:.0f}%"
        }
    }
