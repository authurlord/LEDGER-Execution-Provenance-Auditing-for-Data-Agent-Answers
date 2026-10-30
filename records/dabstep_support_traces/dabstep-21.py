import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Group by merchant and calculate the average transaction amount
avg_amounts = payments.groupby('merchant')['eur_amount'].mean()

# Find the merchant with the highest average transaction amount
highest_avg_merchant = avg_amounts.idxmax()

# Print the result as JSON
print(json.dumps({'answer': highest_avg_merchant}))