import pandas as pd
import json

# Load fees.json
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# Convert to DataFrame for easier filtering
df_fees = pd.DataFrame(fees_data)

# Filter for GlobalCard
df_globalcard = df_fees[df_fees['card_scheme'] == 'GlobalCard']

# Filter for Credit transactions
# We need to handle is_credit being True or potentially None if it applies to all.
# However, the question asks specifically for "credit transactions".
# Usually, if is_credit is null, it might apply to both. 
# Let's look for explicit True first. If none, we might need to consider None.
# But let's look at the sample: ID 2 is GlobalCard, is_credit: null.
# If there are no is_credit: True entries, we might have to use the null ones.
# Let's assume we take entries where is_credit is True OR is_credit is None.

# Actually, let's look at the data more carefully.
# If is_credit is None, it often means "not specified" or "applies to all".
# If the question asks for credit, and there is a specific credit rate, we use it.
# If there is only a general rate (null), we use it.

# Let's filter for is_credit == True or is_credit.isna()
df_credit = df_globalcard[df_globalcard['is_credit'].isin([True]) | df_globalcard['is_credit'].isna()]

# Calculate fee for 10 EUR
# Formula: fixed_amount + (eur_amount * rate / 10000)
eur_amount = 10.0

# Check if 'rate' and 'fixed_amount' exist and are numeric
df_credit = df_credit.dropna(subset=['rate', 'fixed_amount'])

# Calculate fee
df_credit['fee'] = df_credit['fixed_amount'] + (eur_amount * df_credit['rate'] / 10000)

# Calculate average fee
if not df_credit.empty:
    avg_fee = df_credit['fee'].mean()
    answer = round(avg_fee, 6)
else:
    answer = 'Not Applicable'

print(json.dumps({'answer': answer}))