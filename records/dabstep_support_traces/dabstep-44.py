import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count total number of shoppers (unique email addresses, excluding nulls)
# We consider a "shopper" as a unique email address
total_shoppers = payments['email_address'].dropna().nunique()

# Count number of repeat customers (email addresses that appear more than once)
# First, get the count of each email address
email_counts = payments['email_address'].dropna().value_counts()

# Repeat customers are those with count > 1
repeat_customers = (email_counts > 1).sum()

# Calculate the percentage of repeat customers
if total_shoppers > 0:
    percentage = (repeat_customers / total_shoppers) * 100
else:
    percentage = 0.0

# Round to 6 decimals
percentage = round(percentage, 6)

# Output the result
print(json.dumps({'answer': percentage}))