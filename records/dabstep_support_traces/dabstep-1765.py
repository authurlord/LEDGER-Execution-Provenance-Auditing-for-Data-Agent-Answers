import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)
with open('fees.json', 'r') as f:
    fees = json.load(f)

# 1. Get merchant info for Crossfit_Hanna
merchant_info = None
for m in merchant_data:
    if m['merchant'] == 'Crossfit_Hanna':
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    mcc = merchant_info['merchant_category_code']
    
    # 2. Filter payments for Crossfit_Hanna in March 2023
    # March 2023: Day of year 60 (Mar 1) to 90 (Mar 31)
    crossfit_march = payments[
        (payments['merchant'] == 'Crossfit_Hanna') &
        (payments['year'] == 2023) &
        (payments['day_of_year'] >= 60) &
        (payments['day_of_year'] <= 90)
    ]
    
    if crossfit_march.empty:
        print(json.dumps({'answer': ''}))
    else:
        # 3. Find applicable fee IDs
        applicable_ids = set()
        
        for _, row in crossfit_march.iterrows():
            card_scheme = row['card_scheme']
            is_credit = row['is_credit']
            aci = row['aci']
            
            for fee in fees:
                # Check card_scheme
                if fee['card_scheme'] != card_scheme:
                    continue
                
                # Check is_credit
                fee_is_credit = fee['is_credit']
                if fee_is_credit is not None and fee_is_credit != is_credit:
                    continue
                
                # Check aci
                if aci not in fee['aci']:
                    continue
                
                # Check merchant_category_code
                if mcc not in fee['merchant_category_code']:
                    continue
                
                applicable_ids.add(fee['ID'])
        
        # Sort the IDs for consistent output
        sorted_ids = sorted(list(applicable_ids))
        answer = ', '.join(map(str, sorted_ids))
        print(json.dumps({'answer': answer}))