import pandas as pd
import json

# Read the payments.csv file
payments_df = pd.read_csv('payments.csv')

# Count the occurrences of each card scheme
card_scheme_counts = payments_df['card_scheme'].value_counts()

# Get the most commonly used card scheme
most_common_card_scheme = card_scheme_counts.idxmax()

# Prepare the answer
answer = most_common_card_scheme

# Print the result in the required format
print(json.dumps({'answer': answer}))