import pandas as pd
import json

# Load merchant data
with open('merchant_data.json', 'r') as f:
    merchants = json.load(f)

# Load fees data
with open('fees.json', 'r') as f:
    fees = json.load(f)

# Load payments data
payments = pd.read_csv('payments.csv')

# 1. Find merchant details for Belles_cookbook_store
merchant_name = "Belles_cookbook_store"
merchant_info = None
for m in merchants:
    if m['merchant'] == merchant_name:
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    mcc = merchant_info['merchant_category_code']
    account_type = merchant_info['account_type']
    
    # 2. Find transactions for this merchant on day 300 of 2023
    # Filter payments
    transactions = payments[
        (payments['merchant'] == merchant_name) &
        (payments['year'] == 2023) &
        (payments['day_of_year'] == 300)
    ]
    
    applicable_fee_ids = set()
    
    if transactions.empty:
        # If no transactions, we cannot determine is_credit or aci.
        # However, the question asks for fees applicable to the store.
        # Usually, this implies looking at the merchant's profile.
        # But fees depend on transaction attributes.
        # Let's check if there are any fees that match the merchant's static attributes
        # and have null/empty conditions for dynamic ones.
        # Or, perhaps the question implies we should look at ALL transactions for the merchant?
        # "For the 300th of the year 2023" is very specific.
        # If there are no transactions, maybe the answer is empty or we look at the merchant's general fees.
        # Let's assume if there are no transactions, we can't determine dynamic fees.
        # But let's first check if there ARE transactions.
        pass
    else:
        # For each transaction, find matching fees
        for idx, row in transactions.iterrows():
            card_scheme = row['card_scheme']
            is_credit = row['is_credit']
            aci = row['aci']
            
            for fee in fees:
                # Check MCC
                if mcc not in fee['merchant_category_code']:
                    continue
                
                # Check Account Type
                if fee['account_type']:
                    if account_type not in fee['account_type']:
                        continue
                # If fee['account_type'] is empty, it applies to all account types? 
                # The sample shows empty list [] for some. Let's assume empty means "any".
                
                # Check Is Credit
                if fee['is_credit'] is not None:
                    if fee['is_credit'] != is_credit:
                        continue
                
                # Check ACI
                if fee['aci']:
                    if aci not in fee['aci']:
                        continue
                
                # Check Card Scheme
                # The fee has a specific card_scheme. Does it match?
                if fee['card_scheme'] != card_scheme:
                    continue
                
                applicable_fee_ids.add(fee['ID'])

    # Convert to sorted list for consistent output
    result = sorted(list(applicable_fee_ids))
    
    # Format as comma-separated string
    if not result:
        answer = ""
    else:
        answer = ", ".join(map(str, result))
        
    print(json.dumps({'answer': answer}))