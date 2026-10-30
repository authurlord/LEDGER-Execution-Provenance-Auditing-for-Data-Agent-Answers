import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count the number of transactions flagged as fraudulent
# The column 'has_fraudulent_dispute' indicates if a transaction was flagged as fraudulent
fraudulent_count = payments['has_fraudulent_dispute'].sum()

# Print the result as JSON
print(json.dumps({'answer': int(fraudulent_count)}))