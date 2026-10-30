import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Calculate the total amount volume
total_amount = payments['eur_amount'].sum()

# Group by merchant and sum the amounts
merchant_amounts = payments.groupby('merchant')['eur_amount'].sum()

# Sort the merchants by amount in descending order
merchant_amounts_sorted = merchant_amounts.sort_values(ascending=False)

# Get the top 2 merchants
top_2_merchants = merchant_amounts_sorted.head(2)

# Calculate the total amount for the top 2 merchants
top_2_amount = top_2_merchants.sum()

# Calculate the percentage of transactions from the top 2 merchants
percentage = (top_2_amount / total_amount) * 100

# Round to 6 decimals
percentage_rounded = round(percentage, 6)

# Print the result as JSON
print(json.dumps({'answer': percentage_rounded}))