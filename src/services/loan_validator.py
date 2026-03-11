"""
Business logic for loan validation and risk assessment
"""
from typing import Dict, List, Any

class LoanValidator:
    """Validates loans against risk rules"""
    
    # Risk thresholds
    MAX_LOAN_TO_INCOME = 30  # percent
    MAX_DTI = 35 # percent
    MIN_ANNUAL_INCOME = 40000  # dollars
    MAX_LOAN_TO_INCOME_RATIO = 0.5  # 50% of annual income
    
    @staticmethod
    def validate_loan(loan_amnt: float, annual_inc: float, dti: float) -> Dict[str, Any]:
        """
        Validate loan application against risk rules
        
        Args:
            loan_amnt: Loan amount in dollars
            annual_inc: Annual income in dollars
            dti: Debt-to-income ratio as percentage (0-100)
            
        Returns:
            Dict with validation results, risk flags, and recommendation
        """
        risk_flags = []
        is_approved = True
        
        # Calculate loan-to-income ratio
        loan_to_income = (loan_amnt / annual_inc) * 100
        
        # Rule 1: Check Loan-to-Income Ratio
        if loan_to_income > LoanValidator.MAX_LOAN_TO_INCOME:
            risk_flags.append(
                f"High Loan-to-Income: {loan_to_income:.1f}% "
                f"(max {LoanValidator.MAX_LOAN_TO_INCOME}%)"
            )
            is_approved = False
        
        # Rule 2: Check DTI Ratio
        if dti > LoanValidator.MAX_DTI:
            risk_flags.append(
                f"High DTI: {dti:.1f}% (max {LoanValidator.MAX_DTI}%)"
            )
            is_approved = False
        
        # Rule 3: Check minimum income requirement
        if annual_inc < LoanValidator.MIN_ANNUAL_INCOME:
            risk_flags.append(
                f"Income too low: ${annual_inc:,.0f} "
                f"(min ${LoanValidator.MIN_ANNUAL_INCOME:,})"
            )
            is_approved = False
        
        # Rule 4: Check loan amount relative to income
        if loan_amnt > annual_inc * LoanValidator.MAX_LOAN_TO_INCOME_RATIO:
            risk_flags.append(
                f"Loan amount exceeds {LoanValidator.MAX_LOAN_TO_INCOME_RATIO * 100:.0f}% of annual income"
            )
            is_approved = False
        
        # Determine risk level
        if not is_approved:
            risk_level = "HIGH"
        elif loan_to_income > 25 or dti > 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "is_approved": is_approved,
            "risk_level": risk_level,
            "loan_to_income_ratio": f"{loan_to_income:.1f}%",
            "risk_flags": risk_flags
        }
    
    @staticmethod
    def apply_override_logic(model_recommendation: str, validation_result: Dict[str, Any]) -> str:
        """
        Apply business logic to override model recommendation if needed
        
        Args:
            model_recommendation: Model's recommendation ('APPROVED' or 'REJECTED')
            validation_result: Risk validation results
            
        Returns:
            Final recommendation ('APPROVED' or 'REJECTED')
        """
        # If validation fails, override model
        if not validation_result["is_approved"]:
            return "REJECTED"
        
        # Otherwise, trust the model
        return model_recommendation
    
    @staticmethod
    def get_all_rules() -> List[Dict[str, Any]]:
        """Get all validation rules for documentation"""
        return [
            {
                "name": "Loan-to-Income Ratio",
                "threshold": f"< {LoanValidator.MAX_LOAN_TO_INCOME}%",
                "description": "Loan amount should not exceed 30% of annual income"
            },
            {
                "name": "Debt-to-Income Ratio",
                "threshold": f"< {LoanValidator.MAX_DTI}%",
                "description": "Total debt should not exceed 35% of income"
            },
            {
                "name": "Minimum Annual Income",
                "threshold": f">= ${LoanValidator.MIN_ANNUAL_INCOME:,}",
                "description": "Applicant must have minimum annual income"
            },
            {
                "name": "Loan Size",
                "threshold": f"<= {LoanValidator.MAX_LOAN_TO_INCOME_RATIO * 100:.0f}% of income",
                "description": "Loan amount should not exceed 50% of annual income"
            }
        ]