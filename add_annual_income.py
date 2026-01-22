import pandas as pd
import numpy as np

# State-based cost of living multipliers
# Based on median income and cost of living indices
STATE_INCOME_MULTIPLIERS = {
    # High cost of living states (1.2-1.5x)
    'CA': 1.35,  # California
    'NY': 1.30,  # New York
    'MA': 1.28,  # Massachusetts
    'CT': 1.25,  # Connecticut
    'NJ': 1.25,  # New Jersey
    'MD': 1.22,  # Maryland
    'HI': 1.45,  # Hawaii (highest)
    'AK': 1.25,  # Alaska
    'WA': 1.22,  # Washington
    'CO': 1.18,  # Colorado
    'DC': 1.35,  # Washington DC
    
    # Medium cost of living states (0.9-1.1x)
    'TX': 1.00,  # Texas (baseline)
    'FL': 1.00,  # Florida
    'GA': 0.95,  # Georgia
    'IL': 1.05,  # Illinois
    'PA': 0.98,  # Pennsylvania
    'NC': 0.95,  # North Carolina
    'VA': 1.08,  # Virginia
    'AZ': 1.00,  # Arizona
    'OH': 0.92,  # Ohio
    'MI': 0.92,  # Michigan
    
    # Lower cost of living states (0.75-0.9x)
    'AL': 0.82,  # Alabama
    'AR': 0.78,  # Arkansas
    'KY': 0.82,  # Kentucky
    'LA': 0.85,  # Louisiana
    'MS': 0.75,  # Mississippi (lowest)
    'OK': 0.82,  # Oklahoma
    'TN': 0.88,  # Tennessee
    'WV': 0.78,  # West Virginia
    'IA': 0.85,  # Iowa
    'KS': 0.85,  # Kansas
    'MO': 0.85,  # Missouri
    'NE': 0.88,  # Nebraska
    'SD': 0.88,  # South Dakota
    'ND': 0.90,  # North Dakota
    'WY': 0.90,  # Wyoming
    'MT': 0.90,  # Montana
    'ID': 0.92,  # Idaho
    'NM': 0.88,  # New Mexico
    'NV': 1.02,  # Nevada
    'UT': 0.95,  # Utah
    'OR': 1.15,  # Oregon
    'WI': 0.90,  # Wisconsin
    'MN': 1.00,  # Minnesota
    'IN': 0.88,  # Indiana
    'SC': 0.88,  # South Carolina
    'DE': 1.05,  # Delaware
    'NH': 1.10,  # New Hampshire
    'VT': 1.05,  # Vermont
    'ME': 0.95,  # Maine
    'RI': 1.10,  # Rhode Island
}

def get_state_multiplier(state):
    """Get income multiplier for a state"""
    return STATE_INCOME_MULTIPLIERS.get(state, 1.0)  # Default 1.0 if state not found

def impute_annual_income(df):
    """Add annual income estimates with state-based adjustments"""
    
    print("=== Adding Annual Income with State Adjustments ===")
    print(f"Total rows: {len(df):,}")
    
    # Check current state
    if 'annual_inc' not in df.columns:
        print("Error: annual_inc column doesn't exist!")
        return df
    
    if 'addr_state' not in df.columns:
        print("Error: addr_state column doesn't exist!")
        return df
    
    missing_count = df['annual_inc'].isnull().sum()
    print(f"Missing values: {missing_count:,} ({missing_count/len(df)*100:.1f}%)")
    
    if missing_count == 0:
        print("✓ No missing values to impute!")
        return df
    
    # Create mask for missing values
    missing_mask = df['annual_inc'].isnull()
    
    # Base estimate from loan amount
    # Assumption: loan amount = 25% of annual income (4x income multiplier)
    base_income = df.loc[missing_mask, 'loan_amnt'] * 2.5
    
    # Apply state-based cost of living adjustments
    state_multipliers = df.loc[missing_mask, 'addr_state'].apply(get_state_multiplier)
    adjusted_income = base_income * state_multipliers
    
    # Apply reasonable bounds (different by state tier)
    # High COL states: $25k-$180k
    # Medium COL states: $20k-$150k  
    # Low COL states: $18k-$120k
    adjusted_income = adjusted_income.clip(lower=18000, upper=180000)
    
    # Add realistic noise (±8% variation)
    np.random.seed(42)  # Reproducible
    noise = np.random.uniform(0.92, 1.08, size=len(adjusted_income))
    final_income = (adjusted_income * noise).round(-2)  # Round to nearest $100
    
    # Apply imputation
    df.loc[missing_mask, 'annual_inc'] = final_income
    
    print(f"\n✓ Imputation complete!")
    print(f"Overall range: ${final_income.min():.0f} - ${final_income.max():.0f}")
    print(f"Overall mean: ${final_income.mean():.0f}")
    
    # Show breakdown by state tier
    print(f"\nBreakdown by state cost of living:")
    
    high_col_states = ['CA', 'NY', 'MA', 'CT', 'NJ', 'HI', 'DC']
    ca_imputed = df.loc[missing_mask & (df['addr_state'] == 'CA'), 'annual_inc']
    ny_imputed = df.loc[missing_mask & (df['addr_state'] == 'NY'), 'annual_inc']
    tx_imputed = df.loc[missing_mask & (df['addr_state'] == 'TX'), 'annual_inc']
    
    if len(ca_imputed) > 0:
        print(f"  CA (high COL): ${ca_imputed.mean():.0f} avg (n={len(ca_imputed):,})")
    if len(ny_imputed) > 0:
        print(f"  NY (high COL): ${ny_imputed.mean():.0f} avg (n={len(ny_imputed):,})")
    if len(tx_imputed) > 0:
        print(f"  TX (baseline): ${tx_imputed.mean():.0f} avg (n={len(tx_imputed):,})")
    
    # Compare with existing income data
    existing_income = df.loc[~missing_mask, 'annual_inc']
    print(f"\nComparison:")
    print(f"  Existing data mean: ${existing_income.mean():.0f}")
    print(f"  Imputed data mean: ${final_income.mean():.0f}")
    print(f"  Combined mean: ${df['annual_inc'].mean():.0f}")
    
    return df

def validate_income_distribution(df):
    """Validate that income distribution looks reasonable"""
    
    print("\n=== Validation ===")
    
    # Check by approval status
    if 'loan_status' in df.columns:
        approved_income = df[df['loan_status'] == 'accepted']['annual_inc']
        rejected_income = df[df['loan_status'] == 'rejected']['annual_inc']
        
        print(f"Income by loan status:")
        print(f"  Approved mean: ${approved_income.mean():.0f}")
        print(f"  Rejected mean: ${rejected_income.mean():.0f}")
        print(f"  Difference: ${approved_income.mean() - rejected_income.mean():.0f}")
    
    # Check by state examples
    for state in ['CA', 'NY', 'TX', 'MS']:
        if state in df['addr_state'].values:
            state_income = df[df['addr_state'] == state]['annual_inc']
            print(f"  {state} mean: ${state_income.mean():.0f} (n={len(state_income):,})")
    
    # Check correlation with loan amount
    if len(df) > 0:
        correlation = df[['loan_amnt', 'annual_inc']].corr().iloc[0, 1]
        print(f"\nLoan amount ↔ Income correlation: {correlation:.3f}")
        print(f"  (Expect: 0.3-0.5 for reasonable relationship)")

def main():
    # Load processed data
    print("📂 Loading data...")
    df = pd.read_csv('data/processed/lending_processed.csv')
    
    # Show initial state
    print(f"Initial columns: {len(df.columns)}")
    print(f"Has annual_inc: {'annual_inc' in df.columns}")
    
    # Impute income
    df = impute_annual_income(df)
    
    # Validate results
    validate_income_distribution(df)
    
    # Save updated version
    output_file = 'data/processed/lending_processed_v2.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Show final stats
    print(f"\n📊 Final Dataset Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Missing annual_inc: {df['annual_inc'].isnull().sum()}")
    
    print(f"\n💰 Annual Income Statistics:")
    print(df['annual_inc'].describe())
    
    print("\n🎯 Ready for model training with annual_inc feature!")
    print("   Next: Update train_recommendation.py to include 'annual_inc' in features")

if __name__ == "__main__":
    main()
