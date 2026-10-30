import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Group by merchant and sum the eur_amount
merchant_volumes = payments.groupby('merchant')['eur_amount'].sum()

# Find the merchant with the highest total transaction amount volume
highest_volume_merchant = merchant_volumes.idxmax()

# Print the result as JSON
print(json.dumps({'answer': highest_volume_merchant}))