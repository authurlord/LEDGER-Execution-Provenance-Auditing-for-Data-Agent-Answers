import pandas as pd
import json

# Load the payments dataset
payments = pd.read_csv('payments.csv')

# Filter for year 2023
payments_2023 = payments[payments['year'] == 2023]

# Filter for Ecommerce and In-store transactions
ecommerce = payments_2023[payments_2023['shopper_interaction'] == 'Ecommerce']
in_store = payments_2023[payments_2023['shopper_interaction'] == 'InStore']

# Calculate fraud rate for Ecommerce
# Fraud rate is defined as the proportion of transactions with a fraudulent dispute
if len(ecommerce) > 0:
    ecommerce_fraud_rate = ecommerce['has_fraudulent_dispute'].mean()
else:
    ecommerce_fraud_rate = 0

# Calculate fraud rate for In-store
if len(in_store) > 0:
    in_store_fraud_rate = in_store['has_fraudulent_dispute'].mean()
else:
    in_store_fraud_rate = 0

# Compare the rates
if ecommerce_fraud_rate > in_store_fraud_rate:
    answer = "yes"
else:
    answer = "no"

print(json.dumps({'answer': answer}))