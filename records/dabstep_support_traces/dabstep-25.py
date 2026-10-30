import pandas as pd
import json
import numpy as np

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Extract the eur_amount column
amounts = payments['eur_amount']

# Calculate mean and standard deviation
mean_amount = amounts.mean()
std_amount = amounts.std()

# Calculate Z-scores
z_scores = (amounts - mean_amount) / std_amount

# Count outliers where absolute Z-score > 3
outliers = (z_scores.abs() > 3).sum()

# Output the result
print(json.dumps({'answer': int(outliers)}))