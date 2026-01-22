import pandas as pd
import numpy as np

# State multipliers
STATE_MULTIPLIERS = {
    'CA': 1.35, 'NY': 1.30, 'MA': 1.28, 'CT': 1.25, 'NJ': 1.25,
    'HI': 1.45, 'WA': 1.22, 'CO': 1.18, 'DC': 1.35,
    'TX': 1.00, 'FL': 1.00, 'AZ': 1.00,
    'AL': 0.82, 'AR': 0.78, 'MS': 0.75, 'OK': 0.82, 'TN': 0.88
}

def get_multiplier(state):
    return STATE_MULTIPLIERS.get(state, 1.0)

print("Loading data...")
df = pd.read_csv('data/processed/lending_processed.csv')

print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Create annual_inc column from scratch
print("\nCreating annual_inc column...")

# Base estimate from loan amount (loan = 25% of income)
base_income = df['loan_amnt'] * 2.5

# Apply state adjustments
state_mult = df['addr_state'].apply(get_multiplier)
adjusted_income = base_income * state_mult

# Apply bounds and noise
adjusted_income = adjusted_income.clip(18000, 180000)
np.random.seed(42)
noise = np.random.uniform(0.92, 1.08, size=len(adjusted_income))
final_income = (adjusted_income * noise).round(-2)

# Add to dataframe
df['annual_inc'] = final_income

print(f"\nCreated annual_inc column!")
print(f"Range: ${df['annual_inc'].min():.0f} - ${df['annual_inc'].max():.0f}")
print(f"Mean: ${df['annual_inc'].mean():.0f}")

# Show by state
print("\nSample by state:")
for state in ['CA', 'NY', 'TX', 'MS']:
    if state in df['addr_state'].values:
        state_income = df[df['addr_state'] == state]['annual_inc']
        print(f"  {state}: ${state_income.mean():.0f} avg")

# Save
output_file = 'data/processed/lending_processed_v2.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")
print(f"New shape: {df.shape}")