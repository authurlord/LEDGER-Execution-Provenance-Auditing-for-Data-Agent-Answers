import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Filter for fraudulent transactions
fraudulent_transactions = payments[payments['has_fraudulent_dispute'] == True]

# Count the occurrences of each device type in fraudulent transactions
device_type_counts = fraudulent_transactions['device_type'].value_counts()

# Find the most common device type
if not device_type_counts.empty:
    most_common_device = device_type_counts.idxmax()
    answer = most_common_device
else:
    answer = 'Not Applicable'

# Print the result as JSON
print(json.dumps({'answer': answer}))