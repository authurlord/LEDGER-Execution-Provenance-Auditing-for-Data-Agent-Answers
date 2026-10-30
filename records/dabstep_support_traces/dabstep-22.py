import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Check if the required columns exist
if 'eur_amount' in payments.columns and 'has_fraudulent_dispute' in payments.columns:
    # Calculate the correlation between eur_amount and has_fraudulent_dispute
    # has_fraudulent_dispute is a boolean, so it will be treated as 0/1 for correlation
    correlation = payments['eur_amount'].corr(payments['has_fraudulent_dispute'])
    
    # Check if the absolute correlation is greater than 0.5
    if abs(correlation) > 0.5:
        answer = "yes"
    else:
        answer = "no"
else:
    answer = "Not Applicable"

# Print the result as JSON
print(json.dumps({'answer': answer}))