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

from services.loan_predictor import LoanPredictor

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
    annual_inc: float = Field(..., ge=10000, le=500000, description="Annual income in dollars")  # ADD THIS
    addr_state: str = Field(..., min_length=2, max_length=2, description="State code")
    zip_code: str = Field(..., description="ZIP code")
    emp_length: Optional[str] = None
    
    @validator('addr_state')
    def uppercase_state(cls, v):
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "loan_amnt": 15000,
                "dti": 18.5,
                "annual_inc": 75000,  # ADD THIS
                "addr_state": "TX",
                "zip_code": "75001",
                "emp_length": "5 years"
            }
        }

@router.post("/predict")
async def predict_loan(request: LoanApplicationRequest):
    """Get loan recommendation"""
    try:
        predictor = get_loan_predictor()
        result = predictor.predict(request.dict())
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
            "model": predictor.metadata['model_name'],
            "features": len(predictor.feature_names)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
