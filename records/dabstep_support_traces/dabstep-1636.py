import pandas as pd
import json

# Load fees.json
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# Filter for TransactPlus
transact_plus_fees = [f for f in fees_data if f.get('card_scheme') == 'TransactPlus']

# Filter for Account Type F
# We assume empty list [] means all account types, so it matches F
# If the list is not empty, it must contain 'F'
applicable_fees = []
for fee in transact_plus_fees:
    account_types = fee.get('account_type', [])
    if not account_types:  # Empty list, matches all
        applicable_fees.append(fee)
    elif 'F' in account_types:
        applicable_fees.append(fee)

# Calculate fees for 1000 EUR
# Rate is in basis points (0.01%)
transaction_value = 1000.0
fees = []
for fee in applicable_fees:
    fixed_amount = fee.get('fixed_amount', 0)
    rate = fee.get('rate', 0)
    # Rate is likely basis points
    calculated_fee = fixed_amount + (transaction_value * rate / 10000)
    fees.append(calculated_fee)

# Calculate average
if fees:
    average_fee = sum(fees) / len(fees)
else:
    average_fee = None

# Output
if average_fee is not None:
    print(json.dumps({'answer': round(average_fee, 6)}))
else:
    print(json.dumps({'answer': 'Not Applicable'}))