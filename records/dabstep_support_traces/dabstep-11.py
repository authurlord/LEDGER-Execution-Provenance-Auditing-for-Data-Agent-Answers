import pandas as pd
import json

# Read the payments.csv file
payments_df = pd.read_csv('payments.csv')

# Find the highest transaction amount in EUR
highest_amount = payments_df['eur_amount'].max()

# Output the result as JSON
print(json.dumps({'answer': highest_amount}))