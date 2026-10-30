import pandas as pd
import json

# Load merchant data
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)

# Load fees data
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# Load payments data
payments = pd.read_csv('payments.csv')

# 1. Get merchant attributes for Crossfit_Hanna
merchant_info = None
for m in merchant_data:
    if m['merchant'] == 'Crossfit_Hanna':
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    merchant_mcc = merchant_info['merchant_category_code']
    merchant_account_type = merchant_info['account_type']
    
    # 2. Filter payments for Crossfit_Hanna on 12th of 2023
    # Note: day_of_year 12 corresponds to Jan 12th.
    relevant_payments = payments[
        (payments['merchant'] == 'Crossfit_Hanna') &
        (payments['year'] == 2023) &
        (payments['day_of_year'] == 12)
    ]
    
    if relevant_payments.empty:
        # If no transactions on that day, we might still consider the merchant's static profile?
        # The question asks "For the 12th of the year 2023", implying a temporal context.
        # If no transactions, no fees are "applicable" in the sense of being charged.
        # However, usually "applicable" means "which fee rules match the merchant/transaction profile".
        # Let's assume if there are no transactions, the answer is empty or we check merchant profile.
        # Given the phrasing, it likely refers to transactions.
        applicable_ids = []
    else:
        applicable_ids = set()
        
        # We need to check each transaction against the fee rules.
        # A fee is applicable if it matches the transaction's attributes.
        # Attributes from transaction: card_scheme, aci, is_credit
        # Attributes from merchant: account_type, merchant_category_code
        
        for _, row in relevant_payments.iterrows():
            tx_card_scheme = row['card_scheme']
            tx_aci = row['aci']
            tx_is_credit = row['is_credit']
            
            for fee in fees_data:
                # Check card_scheme
                if fee['card_scheme'] != tx_card_scheme:
                    continue
                
                # Check account_type (list contains merchant's account type)
                if merchant_account_type not in fee['account_type']:
                    continue
                
                # Check merchant_category_code (list contains merchant's MCC)
                if merchant_mcc not in fee['merchant_category_code']:
                    continue
                
                # Check is_credit
                fee_is_credit = fee['is_credit']
                if fee_is_credit is not None and fee_is_credit != tx_is_credit:
                    continue
                
                # Check aci (list contains transaction's aci)
                if tx_aci not in fee['aci']:
                    continue
                
                # If all conditions met, this fee ID is applicable
                applicable_ids.add(fee['ID'])
                
    # Convert set to sorted list for consistent output
    answer = sorted(list(applicable_ids))
    
    # Format answer as comma separated string
    if not answer:
        ans_str = ""
    else:
        ans_str = ", ".join(map(str, answer))
        
    print(json.dumps({'answer': ans_str}))