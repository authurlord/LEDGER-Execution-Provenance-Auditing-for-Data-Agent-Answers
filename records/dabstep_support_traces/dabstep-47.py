import pandas as pd
import numpy as np
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Step 1: Calculate the 90th percentile of eur_amount
p90 = payments['eur_amount'].quantile(0.90)

# Step 2: Identify high-value transactions (above the 90th percentile)
high_value_transactions = payments[payments['eur_amount'] > p90]

# Step 3: Identify repeat customers
# A repeat customer is defined as an email_address that appears more than once in the entire dataset.
# We drop null emails for counting purposes as they can't be identified as repeat.
email_counts = payments['email_address'].dropna().value_counts()
repeat_customers = email_counts[email_counts > 1].index

# Step 4: Filter high-value transactions for those made by repeat customers
# We need to check if the email_address of the high-value transaction is in the repeat_customers list.
# Note: If email_address is null, it's not in repeat_customers.
high_value_with_email = high_value_transactions.dropna(subset=['email_address'])
repeat_customer_high_value = high_value_with_email[high_value_with_email['email_address'].isin(repeat_customers)]

# Step 5: Calculate the percentage
# The question asks for the percentage of high-value transactions made by repeat customers.
# Total high-value transactions is len(high_value_transactions).
# High-value transactions by repeat customers is len(repeat_customer_high_value).

total_high_value = len(high_value_transactions)
repeat_high_value = len(repeat_customer_high_value)

if total_high_value == 0:
    answer = 'Not Applicable'
else:
    percentage = (repeat_high_value / total_high_value) * 100
    # Round to 3 decimal places
    answer = round(percentage, 3)

# Output the result
print(json.dumps({'answer': answer}))