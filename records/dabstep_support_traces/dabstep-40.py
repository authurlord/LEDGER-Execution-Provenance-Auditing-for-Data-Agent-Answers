import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Calculate the 90th percentile of eur_amount
p90 = payments['eur_amount'].quantile(0.90)

# Count the number of transactions above the 90th percentile
high_value_count = (payments['eur_amount'] > p90).sum()

# Total number of transactions
total_count = len(payments)

# Calculate the percentage
percentage = (high_value_count / total_count) * 100

# Round to 6 decimals
answer = round(percentage, 6)

# Print the result as JSON
print(json.dumps({'answer': answer}))