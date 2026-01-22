import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_lending_data():
    """Load Lending Club accepted and rejected loan data"""
    raw_data_path = Path("data/raw/archive")
    
    print("=== Loading Lending Club Data ===")
    print(f"Looking for data files in: {raw_data_path}")
    
    # Files are in subdirectories based on your structure
    accepted_file = raw_data_path / "accepted.csv" / "accepted_2007_to_2018Q4.csv"
    rejected_file = raw_data_path / "rejected.csv" / "rejected_2007_to_2018Q4.csv"
    
    if not accepted_file.exists():
        print(f"Error: {accepted_file} not found!")
        print(f"Checked path: {accepted_file.absolute()}")
        return None, None
    
    if not rejected_file.exists():
        print(f"Error: {rejected_file} not found!")
        print(f"Checked path: {rejected_file.absolute()}")
        return None, None
    
    print(f"Found accepted file: {accepted_file}")
    print(f"Found rejected file: {rejected_file}")
    
    # Load just the first few rows to understand structure
    print("Analyzing file structures...")
    accepted_sample = pd.read_csv(accepted_file, nrows=5, low_memory=False)
    rejected_sample = pd.read_csv(rejected_file, nrows=5, low_memory=False)
    
    print(f"Accepted file columns ({len(accepted_sample.columns)}): {list(accepted_sample.columns)}")
    print(f"Rejected file columns ({len(rejected_sample.columns)}): {list(rejected_sample.columns)}")
    
    # Load the full datasets with proper settings
    print("Loading full datasets...")
    accepted_df = pd.read_csv(accepted_file, low_memory=False, dtype=str)  # Load as strings first
    rejected_df = pd.read_csv(rejected_file, low_memory=False, dtype=str)
    
    print(f"Accepted loans shape: {accepted_df.shape}")
    print(f"Rejected loans shape: {rejected_df.shape}")
    
    return accepted_df, rejected_df

def standardize_columns(accepted_df, rejected_df):
    """Standardize column names between accepted and rejected data"""
    
    print("\n=== Standardizing Column Names ===")
    
    # Common Lending Club column mappings
    rejected_mappings = {
        'Amount Requested': 'loan_amnt',
        'Application Date': 'application_date',
        'Loan Title': 'title',
        'Risk_Score': 'fico_range_low',
        'Debt-To-Income Ratio': 'dti',
        'Zip Code': 'zip_code',
        'State': 'addr_state',
        'Employment Length': 'emp_length',
        'Policy Code': 'policy_code'
    }
    
    # Apply mappings to rejected data
    rejected_standardized = rejected_df.copy()
    for old_col, new_col in rejected_mappings.items():
        if old_col in rejected_standardized.columns:
            rejected_standardized = rejected_standardized.rename(columns={old_col: new_col})
            print(f"Mapped: '{old_col}' -> '{new_col}'")
    
    # Clean DTI in rejected data (remove % signs)
    if 'dti' in rejected_standardized.columns:
        print(f"Cleaning DTI column (removing % signs)...")
        rejected_standardized['dti'] = rejected_standardized['dti'].astype(str).str.replace('%', '').str.strip()
    
    # Convert numeric columns in rejected data
    numeric_cols = ['loan_amnt', 'dti', 'fico_range_low']
    for col in numeric_cols:
        if col in rejected_standardized.columns:
            rejected_standardized[col] = pd.to_numeric(rejected_standardized[col], errors='coerce')
            print(f"Converted {col} to numeric")
    
    print(f"Rejected columns after standardization: {list(rejected_standardized.columns)}")
    
    return accepted_df, rejected_standardized
    
    return accepted_df, rejected_standardized

def create_unified_dataset(accepted_df, rejected_df):
    """Create a unified dataset with tiered feature approach"""
    
    print("\n=== Creating Unified Dataset with Tiered Features ===")
    
    # CRITICAL DEBUG: Check input before processing
    print(f"DEBUG - Input accepted_df shape: {accepted_df.shape}")
    print(f"DEBUG - Input rejected_df shape: {rejected_df.shape}")
    print(f"DEBUG - Accepted columns sample: {list(accepted_df.columns[:5])}")
    print(f"DEBUG - Rejected columns sample: {list(rejected_df.columns[:5])}")
    
    # Find overlapping columns
    accepted_cols = set(accepted_df.columns)
    rejected_cols = set(rejected_df.columns)
    common_cols = accepted_cols & rejected_cols
    
    print(f"Common columns found: {len(common_cols)}")
    print(f"Common columns list: {sorted(list(common_cols))}")
    
    # Tier 1: Essential features (must exist in both)
    tier1_features = ['loan_amnt', 'dti']
    tier1_available = [col for col in tier1_features if col in common_cols]
    
    # Tier 2: Valuable common features (if available)
    tier2_features = ['addr_state', 'zip_code', 'emp_length', 'annual_inc']
    tier2_available = [col for col in tier2_features if col in common_cols]
    
    # Tier 3: Accepted-only features (for richer modeling)
    tier3_features = ['grade', 'int_rate', 'term', 'fico_range_low', 'purpose']
    tier3_available = [col for col in tier3_features if col in accepted_cols]
    
    print(f"Tier 1 (Essential): {tier1_available}")
    print(f"Tier 2 (Common valuable): {tier2_available}")  
    print(f"Tier 3 (Accepted-only): {tier3_available}")
    
    if len(tier1_available) < 1:
        print("ERROR: No essential features found.")
        print("DEBUG - This means 'loan_amnt' and 'dti' are not in common_cols")
        print("DEBUG - Accepted has loan_amnt?", 'loan_amnt' in accepted_cols)
        print("DEBUG - Rejected has loan_amnt?", 'loan_amnt' in rejected_cols)
        print("DEBUG - Accepted has dti?", 'dti' in accepted_cols)
        print("DEBUG - Rejected has dti?", 'dti' in rejected_cols)
        return None
    
    # Create base dataset with common features
    base_features = tier1_available + tier2_available
    print(f"DEBUG - Base features to extract: {base_features}")
    
    # PREVENT DUPLICATION: Use .loc to ensure we get a proper copy
    accepted_base = accepted_df.loc[:, base_features].copy()
    rejected_base = rejected_df.loc[:, base_features].copy()
    
    print(f"DEBUG - After base extraction:")
    print(f"  accepted_base shape: {accepted_base.shape}")
    print(f"  rejected_base shape: {rejected_base.shape}")
    
    # Add tier 3 features to accepted data (NaN for rejected)
    for feature in tier3_available:
        if feature in accepted_df.columns:
            accepted_base[feature] = accepted_df[feature].values
            rejected_base[feature] = np.nan
    
    print(f"DEBUG - After tier 3 features:")
    print(f"  accepted_base shape: {accepted_base.shape}")
    print(f"  rejected_base shape: {rejected_base.shape}")
    
    # Add status labels
    accepted_base['loan_status'] = 'accepted'
    rejected_base['loan_status'] = 'rejected'
    
    print(f"DEBUG - After adding loan_status:")
    print(f"  accepted_base shape: {accepted_base.shape}")
    print(f"  rejected_base shape: {rejected_base.shape}")
    print(f"  accepted_base loan_status unique: {accepted_base['loan_status'].unique()}")
    print(f"  rejected_base loan_status unique: {rejected_base['loan_status'].unique()}")
    print(f"  accepted_base columns: {list(accepted_base.columns)}")
    print(f"  rejected_base columns: {list(rejected_base.columns)}")
    
    # Sample for performance if needed
    if len(accepted_base) > 100000:
        print("Sampling accepted loans...")
        accepted_base = accepted_base.sample(n=100000, random_state=42)
    
    if len(rejected_base) > 100000:
        print("Sampling rejected loans...")
        rejected_base = rejected_base.sample(n=100000, random_state=42)
    
    print(f"DEBUG - Before concat:")
    print(f"  accepted_base shape: {accepted_base.shape}")
    print(f"  rejected_base shape: {rejected_base.shape}")
    
    # Combine datasets
    combined_df = pd.concat([accepted_base, rejected_base], ignore_index=True)
    
    print(f"DEBUG - After concat:")
    print(f"  combined_df shape: {combined_df.shape}")
    print(f"  combined_df columns: {list(combined_df.columns)}")
    print(f"  combined_df head:\n{combined_df.head()}")
    
    return combined_df

def clean_lending_data(df):
    """Clean the combined lending data"""
    
    if df is None:
        return None
    
    print("\n=== Cleaning Lending Data ===")
    print(f"Initial shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")
    
    # Remove exact duplicates first
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    if len(df) < 10:
        print(f"WARNING: Only {len(df)} rows remain after deduplication!")
        return df
    
    # Clean loan_amnt
    if 'loan_amnt' in df.columns:
        df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
        # Remove invalid loan amounts
        invalid_loans = (df['loan_amnt'] <= 0) | (df['loan_amnt'] > 50000)
        print(f"Removing {invalid_loans.sum()} rows with invalid loan amounts (<=0 or >$50k)")
        df = df[~invalid_loans]
    
    # Clean DTI - this is critical!
    if 'dti' in df.columns:
        df['dti'] = pd.to_numeric(df['dti'], errors='coerce')
        
        # DTI should be between 0 and 100 (it's a percentage)
        print(f"DTI stats before cleaning: min={df['dti'].min()}, max={df['dti'].max()}, mean={df['dti'].mean():.2f}")
        
        # Remove clearly invalid DTI values
        invalid_dti = (df['dti'] < 0) | (df['dti'] > 100)
        print(f"Removing {invalid_dti.sum()} rows with invalid DTI (<0 or >100)")
        df = df[~invalid_dti]
        
        print(f"DTI stats after cleaning: min={df['dti'].min()}, max={df['dti'].max()}, mean={df['dti'].mean():.2f}")
    
    # Clean FICO scores
    if 'fico_range_low' in df.columns:
        df['fico_range_low'] = pd.to_numeric(df['fico_range_low'], errors='coerce')
        # Valid FICO range is 300-850
        invalid_fico = (df['fico_range_low'] < 300) | (df['fico_range_low'] > 850)
        if invalid_fico.sum() > 0:
            print(f"Setting {invalid_fico.sum()} invalid FICO scores to NaN")
            df.loc[invalid_fico, 'fico_range_low'] = np.nan
    
    # Clean interest rate
    if 'int_rate' in df.columns:
        df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
        # Interest rates should be between 3% and 40%
        invalid_rate = (df['int_rate'] < 3) | (df['int_rate'] > 40)
        if invalid_rate.sum() > 0:
            print(f"Setting {invalid_rate.sum()} invalid interest rates to NaN")
            df.loc[invalid_rate, 'int_rate'] = np.nan
    
    # Check for missing values
    missing_info = df.isnull().sum()
    print(f"\nMissing values per column:")
    for col, missing_count in missing_info.items():
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
    
    # Handle missing values in critical columns
    critical_cols = ['loan_amnt', 'dti', 'loan_status']
    available_critical = [col for col in critical_cols if col in df.columns]
    
    if available_critical:
        before_dropna = len(df)
        df = df.dropna(subset=available_critical)
        removed_rows = before_dropna - len(df)
        print(f"\nRemoved {removed_rows} rows with missing critical values")
    
    # Create binary target
    if 'loan_status' in df.columns:
        df['recommended'] = (df['loan_status'] == 'accepted').astype(int)
        target_counts = df['recommended'].value_counts()
        print(f"Target distribution: {target_counts.to_dict()}")
    
    print(f"\nFinal cleaned shape: {df.shape}")
    return df

def add_basic_features(df):
    """Add basic engineered features"""
    
    if df is None or len(df) < 10:
        return df
    
    print("\n=== Adding Basic Features ===")
    
    # Only add features if we have the required base columns
    if 'loan_amnt' in df.columns:
        # Convert to numeric if needed
        df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
        
        # Create loan size categories
        df['loan_category'] = pd.cut(df['loan_amnt'], 
                                   bins=[0, 5000, 15000, 25000, float('inf')],
                                   labels=['small', 'medium', 'large', 'jumbo'])
        print("Added loan_category feature")
    
    if 'dti' in df.columns:
        # Convert to numeric, handling strings like "15.5%", "N/A", etc.
        print(f"DEBUG - dti column dtype before conversion: {df['dti'].dtype}")
        print(f"DEBUG - dti sample values: {df['dti'].head()}")
        
        # Remove % signs if present and convert to numeric
        if df['dti'].dtype == 'object':
            df['dti'] = df['dti'].astype(str).str.replace('%', '').str.strip()
        
        df['dti'] = pd.to_numeric(df['dti'], errors='coerce')
        print(f"DEBUG - dti column dtype after conversion: {df['dti'].dtype}")
        print(f"DEBUG - dti null count after conversion: {df['dti'].isnull().sum()}")
        
        # Only create categories if we have valid numeric data
        valid_dti = df['dti'].notna()
        if valid_dti.sum() > 0:
            df['dti_risk'] = pd.cut(df['dti'], 
                                   bins=[0, 10, 20, 30, float('inf')],
                                   labels=['low', 'medium', 'high', 'very_high'])
            print(f"Added dti_risk feature ({valid_dti.sum()} valid values)")
        else:
            print("WARNING: No valid DTI values found, skipping dti_risk feature")
    
    return df

def save_processed_data(df, filename="lending_processed.csv"):
    """Save the processed data"""
    
    if df is None:
        print("No data to save!")
        return
    
    processed_data_path = Path("data/processed")
    processed_data_path.mkdir(exist_ok=True)
    
    output_file = processed_data_path / filename
    df.to_csv(output_file, index=False)
    
    print(f"\n=== Data Saved Successfully ===")
    print(f"File: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'recommended' in df.columns:
        target_dist = df['recommended'].value_counts()
        print(f"Target distribution: Accepted={target_dist.get(1, 0)}, Rejected={target_dist.get(0, 0)}")

def main():
    """Main preprocessing pipeline"""
    
    print("🏦 LENDING CLUB RECOMMENDATION DATA PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Load the raw data
    accepted_df, rejected_df = load_lending_data()
    
    if accepted_df is None or rejected_df is None:
        print("❌ Failed to load data files")
        return
    
    # Step 2: Standardize column names
    accepted_df, rejected_df = standardize_columns(accepted_df, rejected_df)
    
    # Step 3: Create unified dataset
    combined_df = create_unified_dataset(accepted_df, rejected_df)
    
    if combined_df is None:
        print("❌ Failed to create unified dataset")
        return
    
    # Step 4: Clean the data
    cleaned_df = clean_lending_data(combined_df)
    
    # Step 5: Add engineered features
    final_df = add_basic_features(cleaned_df)
    
    # Step 6: Save the results
    save_processed_data(final_df)
    
    if final_df is not None and len(final_df) > 100:
        print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("Your data is ready for machine learning model training.")
    else:
        print("\n⚠️  PREPROCESSING COMPLETED WITH ISSUES")
        print("The resulting dataset is very small. Check your input files.")

if __name__ == "__main__":
    main()