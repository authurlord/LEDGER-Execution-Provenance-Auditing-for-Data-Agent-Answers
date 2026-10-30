import pandas as pd
import json
import os

# Load data
with open('fees.json', 'r') as f:
    fees = json.load(f)

with open('merchant_data.json', 'r') as f:
    merchants = json.load(f)

payments = pd.read_csv('payments.csv')

# Find fee ID 384
fee_384 = None
for fee in fees:
    if fee['ID'] == 384:
        fee_384 = fee
        break

if fee_384 is None:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

# Get merchant data for Belles_cookbook_store
merchant_info = None
for m in merchants:
    if m['merchant'] == 'Belles_cookbook_store':
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

merchant_acct_type = merchant_info['account_type']
merchant_mcc = merchant_info['merchant_category_code']

# Filter payments for Belles_cookbook_store in July 2023
# July 2023: Year 2023, Day of Year 182 to 212 (31 days in Jan, 28 in Feb, 31 Mar, 30 Apr, 31 May, 30 Jun = 181 days before July)
# July 1 is day 182, July 31 is day 212.
july_start = 182
july_end = 212

belles_payments = payments[
    (payments['merchant'] == 'Belles_cookbook_store') &
    (payments['year'] == 2023) &
    (payments['day_of_year'] >= july_start) &
    (payments['day_of_year'] <= july_end)
]

# Function to check if a transaction matches the fee criteria
def matches_fee(row, fee):
    # Check card_scheme
    if fee['card_scheme'] and row['card_scheme'] != fee['card_scheme']:
        return False
    
    # Check account_type
    # The fee might have account_type list. Merchant has one account_type.
    if fee['account_type']:
        if merchant_acct_type not in fee['account_type']:
            return False
    
    # Check merchant_category_code
    if fee['merchant_category_code']:
        if merchant_mcc not in fee['merchant_category_code']:
            return False
    
    # Check is_credit
    if fee['is_credit'] is not None:
        if row['is_credit'] != fee['is_credit']:
            return False
    
    # Check aci
    if fee['aci']:
        if row['aci'] not in fee['aci']:
            return False
            
    # Check intracountry
    if fee['intracountry'] is not None:
        # Intracountry usually means domestic transaction.
        # We assume domestic if issuing_country == acquirer_country
        if row['issuing_country'] != row['acquirer_country']:
            return False
            
    return True

# Calculate delta
# Delta = Sum( New_Fee - Old_Fee )
# New_Fee = fixed + eur * 1 / 100
# Old_Fee = fixed + eur * old_rate / 100
# Delta = Sum( eur * (1 - old_rate) / 100 )

old_rate = fee_384['rate']
fixed_amount = fee_384['fixed_amount']

total_delta = 0.0
applicable_transactions = []

for idx, row in belles_payments.iterrows():
    if matches_fee(row, fee_384):
        applicable_transactions.append(row)
        eur_amount = row['eur_amount']
        # Delta for this transaction
        delta = eur_amount * (1 - old_rate) / 100
        total_delta += delta

# If no transactions match, delta is 0? Or Not Applicable?
# The question asks "what delta would ... pay". If no transactions, delta is 0.
# But let's check if there are any transactions at all.
if len(belles_payments) == 0:
    print(json.dumps({'answer': 0.00000000000000}))
else:
    # Round to 14 decimals
    answer = round(total_delta, 14)
    print(json.dumps({'answer': answer}))