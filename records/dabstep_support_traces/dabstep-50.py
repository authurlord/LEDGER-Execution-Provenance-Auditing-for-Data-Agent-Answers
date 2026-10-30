import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Calculate the fraud rate for credit card payments
credit_payments = payments[payments['is_credit'] == True]
if len(credit_payments) > 0:
    credit_fraud_rate = credit_payments['has_fraudulent_dispute'].mean()
else:
    credit_fraud_rate = 0

# Calculate the fraud rate for debit card payments
debit_payments = payments[payments['is_credit'] == False]
if len(debit_payments) > 0:
    debit_fraud_rate = debit_payments['has_fraudulent_dispute'].mean()
else:
    debit_fraud_rate = 0

# Compare the fraud rates
if credit_fraud_rate > debit_fraud_rate:
    answer = "yes"
else:
    answer = "no"

# Output the result
print(json.dumps({'answer': answer}))