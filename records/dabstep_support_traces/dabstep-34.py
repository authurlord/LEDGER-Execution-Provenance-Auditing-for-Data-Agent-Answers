import pandas as pd
import json

# Read the payments.csv file
payments_df = pd.read_csv('payments.csv')

# Count unique IP addresses
unique_shoppers = payments_df['ip_address'].nunique()

# Output the result as JSON
print(json.dumps({'answer': unique_shoppers}))