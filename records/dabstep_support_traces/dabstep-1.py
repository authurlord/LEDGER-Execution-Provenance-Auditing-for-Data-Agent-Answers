import pandas as pd
import json

# Read the payments.csv file which contains the transaction data
payments_df = pd.read_csv('payments.csv')

# Count the total number of transactions
total_transactions = len(payments_df)

# Output the result as JSON
print(json.dumps({'answer': total_transactions}))