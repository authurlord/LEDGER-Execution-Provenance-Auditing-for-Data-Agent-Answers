import pandas as pd
import json

# Read the payments.csv file to get the merchant column
payments_df = pd.read_csv('payments.csv')

# Count unique merchants
unique_merchants = payments_df['merchant'].nunique()

# Output the result as JSON
print(json.dumps({'answer': unique_merchants}))