import pandas as pd
import json
import numpy as np

# Read the payments.csv file
payments = pd.read_csv('payments.csv')

# Calculate Z-Score for eur_amount
# Z-Score = (x - mean) / std
mean_amount = payments['eur_amount'].mean()
std_amount = payments['eur_amount'].std()

# Calculate Z-Scores
payments['z_score'] = (payments['eur_amount'] - mean_amount) / std_amount

# Identify outliers where Z-Score > 3 (absolute value or just > 3? Usually Z-Score > 3 implies magnitude, but let's check the question: "Z-Score > 3". This usually means the absolute value is greater than 3, i.e., |Z| > 3. However, strictly speaking, Z > 3 means only the upper tail. Let's assume |Z| > 3 as is standard for outlier detection unless specified otherwise. But the question says "Z-Score > 3". Let's stick to the literal interpretation first: Z > 3. If that yields no results or seems wrong, I'll reconsider. Actually, in many contexts, "outlier using Z-Score > 3" implies |Z| > 3. Let's calculate both and see. Given the phrasing "Z-Score > 3", it might strictly mean positive outliers. Let's look at the data. If I use |Z| > 3, it's more common. Let's try |Z| > 3 first as it's the standard definition of an outlier in this context. If the question meant strictly > 3, it would likely say "positive Z-Score greater than 3". Let's assume |Z| > 3.)

# Let's re-read carefully: "outlier transactions (using Z-Score > 3)". This is ambiguous. 
# Standard statistical practice: |Z| > 3.
# Literal interpretation: Z > 3.
# Let's check if there are any transactions with Z > 3.
# Let's compute counts for both.

# Option 1: |Z| > 3
outliers_abs = payments[np.abs(payments['z_score']) > 3]

# Option 2: Z > 3
outliers_pos = payments[payments['z_score'] > 3]

# Let's count outliers per merchant for both cases
if len(outliers_abs) > 0:
    outlier_counts_abs = outliers_abs.groupby('merchant').size().reset_index(name='count')
    max_merchant_abs = outlier_counts_abs.loc[outlier_counts_abs['count'].idxmax(), 'merchant']
    max_count_abs = outlier_counts_abs['count'].max()
else:
    max_merchant_abs = None
    max_count_abs = 0

if len(outliers_pos) > 0:
    outlier_counts_pos = outliers_pos.groupby('merchant').size().reset_index(name='count')
    max_merchant_pos = outlier_counts_pos.loc[outlier_counts_pos['count'].idxmax(), 'merchant']
    max_count_pos = outlier_counts_pos['count'].max()
else:
    max_merchant_pos = None
    max_count_pos = 0

# Determine which interpretation to use. 
# If |Z| > 3 gives a result, it's the most likely intended meaning for "outlier".
# If Z > 3 gives no results, then |Z| > 3 is definitely the way.
# If both give results, |Z| > 3 is more standard for "outlier".

if max_merchant_abs is not None:
    answer = max_merchant_abs
else:
    answer = max_merchant_pos

print(json.dumps({'answer': answer}))