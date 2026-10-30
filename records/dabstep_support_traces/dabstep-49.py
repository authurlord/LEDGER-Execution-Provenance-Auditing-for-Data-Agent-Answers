import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Filter for fraudulent transactions
fraudulent_payments = payments[payments['has_fraudulent_dispute'] == True]

# Group by ip_country and count the number of fraudulent transactions
fraud_by_country = fraudulent_payments.groupby('ip_country').size().reset_index(name='count')

# Find the country with the highest number of fraudulent transactions
top_fraud_country = fraud_by_country.loc[fraud_by_country['count'].idxmax(), 'ip_country']

# Check which option matches the top fraud country
options = {
    'A': 'NL',
    'B': 'BE',
    'C': 'ES',
    'D': 'FR'
}

answer = None
for key, value in options.items():
    if value == top_fraud_country:
        answer = f"{key}. {value}"
        break

# If no match found, we still need to provide an answer based on the data
if answer is None:
    # In case the top country is not in the options, we should still report it or handle appropriately
    # But the question asks to choose from A, B, C, D. Let's assume one of them is correct.
    # If not, we might need to pick the closest or report N/A. However, typically these questions have a valid option.
    # Let's just output the top country if it's not in the options, but formatted as requested if possible.
    # Since the instruction says "If a question does not have a relevant or applicable answer... respond with 'Not Applicable'"
    # But here we have a top country. Let's see if it's in the options.
    # If not, we can't really choose A, B, C, or D. So we might need to say Not Applicable.
    # However, let's first check if the top country is indeed one of the options.
    if top_fraud_country in options.values():
        for key, value in options.items():
            if value == top_fraud_country:
                answer = f"{key}. {value}"
                break
    else:
        answer = "Not Applicable"

print(json.dumps({'answer': answer}))