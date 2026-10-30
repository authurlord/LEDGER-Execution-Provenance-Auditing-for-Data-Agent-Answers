import pandas as pd
import json

# Read the payments.csv file
df = pd.read_csv('payments.csv')

# Calculate the number of missing values for each column
missing_counts = df.isnull().sum()

# Find the column with the most missing data
max_missing_column = missing_counts.idxmax()
max_missing_count = missing_counts.max()

# If there are no missing values, handle appropriately
if max_missing_count == 0:
    answer = 'Not Applicable'
else:
    answer = max_missing_column

# Print the result as JSON
print(json.dumps({'answer': answer}))