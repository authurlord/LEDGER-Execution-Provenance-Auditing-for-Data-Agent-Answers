import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count the number of transactions per email address
# Drop null email addresses as they don't represent a specific shopper
email_counts = payments['email_address'].dropna().value_counts()

# Find shoppers who have made more than one transaction
shoppers_with_multiple_transactions = email_counts[email_counts > 1]

# Get the count of such shoppers
num_shoppers = len(shoppers_with_multiple_transactions)

# Output the result
print(json.dumps({'answer': num_shoppers}))