import pandas as pd
import json

# Read the payments.csv file
payments_df = pd.read_csv('payments.csv')

# Calculate the percentage of transactions with an associated email address
# Count total transactions
total_transactions = len(payments_df)

# Count transactions with non-null email addresses
transactions_with_email = payments_df['email_address'].notna().sum()

# Calculate percentage
percentage = (transactions_with_email / total_transactions) * 100

# Round to 6 decimals
percentage_rounded = round(percentage, 6)

# Output the result
print(json.dumps({'answer': percentage_rounded}))