import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count the number of transactions per merchant
merchant_transaction_counts = payments['merchant'].value_counts()

# Find the merchant with the highest number of transactions
top_merchant = merchant_transaction_counts.idxmax()

# Output the result as JSON
print(json.dumps({'answer': top_merchant}))