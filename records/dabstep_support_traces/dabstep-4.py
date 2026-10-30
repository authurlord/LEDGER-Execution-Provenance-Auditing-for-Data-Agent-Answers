import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count the number of transactions per ip_country
country_counts = payments['ip_country'].value_counts()

# Get the country with the highest number of transactions
highest_country = country_counts.idxmax()

# Output the result as JSON
print(json.dumps({'answer': highest_country}))