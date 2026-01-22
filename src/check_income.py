import pandas as pd
df = pd.read_csv('data/processed/lending_processed.csv')
print('Columns in processed data:')
print(df.columns.tolist())
print('\nDoes annual_inc exist?', 'annual_inc' in df.columns)
if 'annual_inc' in df.columns:
    print('\nAnnual income stats:')
    print(df['annual_inc'].describe())
