import pandas as pd
import json

# Read the payments.csv file
payments_df = pd.read_csv('payments.csv')

# Calculate the average transaction amount in EUR
average_amount = payments_df['eur_amount'].mean()

# Round to 3 decimals
average_amount_rounded = round(average_amount, 3)

# Print the result as JSON
print(json.dumps({'answer': average_amount_rounded}))