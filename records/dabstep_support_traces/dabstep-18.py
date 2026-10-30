import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Filter for the year 2023
payments_2023 = payments[payments['year'] == 2023]

# Calculate the fraud rate for each card_scheme
# Fraud rate = number of fraudulent disputes / total transactions
fraud_rates = payments_2023.groupby('card_scheme').agg(
    total_transactions=('psp_reference', 'count'),
    total_fraudulent_disputes=('has_fraudulent_dispute', 'sum')
).reset_index()

# Calculate fraud rate
fraud_rates['fraud_rate'] = fraud_rates['total_fraudulent_disputes'] / fraud_rates['total_transactions']

# Find the card_scheme with the highest fraud rate
highest_fraud_rate_scheme = fraud_rates.loc[fraud_rates['fraud_rate'].idxmax(), 'card_scheme']

# Output the result
print(json.dumps({'answer': highest_fraud_rate_scheme}))