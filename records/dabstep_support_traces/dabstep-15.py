import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Filter transactions for the year 2023
transactions_2023 = payments[payments['year'] == 2023]

# Calculate the total number of transactions in 2023
total_transactions_2023 = len(transactions_2023)

# Calculate the number of fraudulent transactions in 2023
# Assuming 'has_fraudulent_dispute' being True indicates a fraudulent transaction
fraudulent_transactions_2023 = transactions_2023[transactions_2023['has_fraudulent_dispute'] == True].shape[0]

# Calculate the percentage of fraudulent transactions
if total_transactions_2023 > 0:
    percentage_fraudulent = (fraudulent_transactions_2023 / total_transactions_2023) * 100
else:
    percentage_fraudulent = 0.0

# Round to 6 decimal places
percentage_fraudulent_rounded = round(percentage_fraudulent, 6)

# Output the result as JSON
print(json.dumps({'answer': percentage_fraudulent_rounded}))