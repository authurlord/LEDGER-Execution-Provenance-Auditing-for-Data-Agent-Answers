import pandas as pd
import json

# Load the payments data
payments_df = pd.read_csv('payments.csv')

# Filter for the year 2023
payments_2023 = payments_df[payments_df['year'] == 2023]

# Group by merchant and calculate the fraud rate for each merchant
# Fraud rate is defined as the number of fraudulent disputes divided by the total number of transactions
merchant_fraud_rates = payments_2023.groupby('merchant').agg(
    total_transactions=('psp_reference', 'count'),
    fraudulent_transactions=('has_fraudulent_dispute', 'sum')
).reset_index()

# Calculate the fraud rate for each merchant
merchant_fraud_rates['fraud_rate'] = merchant_fraud_rates['fraudulent_transactions'] / merchant_fraud_rates['total_transactions']

# Find the minimum fraud rate
min_fraud_rate = merchant_fraud_rates['fraud_rate'].min()

# Convert to percentage and round to 6 decimal places
min_fraud_rate_percentage = round(min_fraud_rate * 100, 6)

# Output the result
print(json.dumps({'answer': min_fraud_rate_percentage}))