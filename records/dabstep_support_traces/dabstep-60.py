import pandas as pd
import json

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Define the segments to analyze
segments = ['merchant', 'issuer_country', 'card_scheme', 'shopper_interaction']

# Map column names if necessary
# From schema: 'issuing_country' is the column for issuer country
# 'merchant', 'card_scheme', 'shopper_interaction' are direct column names

# Calculate fraud rate for each segment
# Fraud rate = number of fraudulent disputes / total transactions

results = {}

for segment in segments:
    if segment == 'issuer_country':
        col = 'issuing_country'
    else:
        col = segment
    
    # Group by the segment
    grouped = payments.groupby(col)
    
    # Calculate fraud rate for each group
    fraud_rates = grouped['has_fraudulent_dispute'].mean()
    
    # Find the segment with the worst (highest) fraud rate
    worst_segment = fraud_rates.idxmax()
    worst_rate = fraud_rates.max()
    
    results[segment] = (worst_segment, worst_rate)

# Determine which segment has the overall worst fraud rate
worst_overall_segment = None
worst_overall_rate = -1

for segment, (worst_value, rate) in results.items():
    if rate > worst_overall_rate:
        worst_overall_rate = rate
        worst_overall_segment = segment

# The question asks for the segment name (e.g., 'merchant', 'issuer_country', etc.)
# that has the worst fraud rate.

answer = worst_overall_segment

print(json.dumps({'answer': answer}))