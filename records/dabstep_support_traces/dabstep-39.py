import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Calculate the 25th percentile of transaction amounts
q25 = payments['eur_amount'].quantile(0.25)

# Count the number of transactions below the 25th percentile
below_q25 = (payments['eur_amount'] < q25).sum()

# Calculate the percentage
total_transactions = len(payments)
percentage = (below_q25 / total_transactions) * 100

# Round to 6 decimals
answer = round(percentage, 6)

# Print the result as JSON
print(json.dumps({'answer': answer}))