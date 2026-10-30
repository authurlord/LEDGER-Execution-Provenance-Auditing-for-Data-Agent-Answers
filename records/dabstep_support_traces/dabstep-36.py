import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Filter out rows where email_address is null/NaN
payments_with_email = payments.dropna(subset=['email_address'])

# Count the number of transactions per unique shopper (email_address)
transactions_per_shopper = payments_with_email.groupby('email_address').size()

# Calculate the average number of transactions per unique shopper
if len(transactions_per_shopper) > 0:
    average_transactions = transactions_per_shopper.mean()
else:
    average_transactions = 0

# Round to 6 decimals
average_transactions = round(average_transactions, 6)

# Output the result
print(json.dumps({'answer': average_transactions}))