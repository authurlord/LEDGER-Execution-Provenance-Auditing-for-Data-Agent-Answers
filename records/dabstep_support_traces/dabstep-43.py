import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Filter out rows where email_address is null
payments_with_email = payments.dropna(subset=['email_address'])

# Group by email_address and calculate the average transaction amount per unique email
# First, sum the amounts per email, then count the number of transactions per email
email_stats = payments_with_email.groupby('email_address').agg(
    total_amount=('eur_amount', 'sum'),
    transaction_count=('eur_amount', 'count')
).reset_index()

# Calculate the average transaction amount per email
email_stats['avg_amount'] = email_stats['total_amount'] / email_stats['transaction_count']

# Now, calculate the overall average of these per-email averages
overall_average = email_stats['avg_amount'].mean()

# Round to 3 decimals
answer = round(overall_average, 3)

# Print the result as JSON
print(json.dumps({'answer': answer}))