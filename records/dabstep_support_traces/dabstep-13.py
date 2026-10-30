import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Count the number of transactions for each hour of the day
hour_counts = payments['hour_of_day'].value_counts()

# Find the hour with the most transactions
most_active_hour = hour_counts.idxmax()

# Output the result as JSON
print(json.dumps({'answer': int(most_active_hour)}))