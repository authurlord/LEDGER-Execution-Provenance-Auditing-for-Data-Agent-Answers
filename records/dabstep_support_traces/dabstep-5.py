import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count the number of transactions per issuing country
issuing_country_counts = payments['issuing_country'].value_counts()

# Get the issuing country with the highest number of transactions
highest_issuing_country = issuing_country_counts.idxmax()

# Prepare the answer
answer = highest_issuing_country

# Print the result as JSON
print(json.dumps({'answer': answer}))