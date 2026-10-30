import pandas as pd
import json
import os

# Load merchant data
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)

# Find Golfclub_Baron_Friso
merchant_info = None
for m in merchant_data:
    if m['merchant'] == 'Golfclub_Baron_Friso':
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    merchant_mcc = merchant_info['merchant_category_code']
    merchant_account_type = merchant_info['account_type']
    
    # Load payments
    payments = pd.read_csv('payments.csv')
    
    # Filter for Golfclub_Baron_Friso in January 2023
    # January 2023: year=2023, day_of_year <= 31
    jan_2023_payments = payments[
        (payments['merchant'] == 'Golfclub_Baron_Friso') &
        (payments['year'] == 2023) &
        (payments['day_of_year'] <= 31)
    ]
    
    if jan_2023_payments.empty:
        print(json.dumps({'answer': ''}))
    else:
        # Load fees
        with open('fees.json', 'r') as f:
            fees = json.load(f)
            
        applicable_fee_ids = set()
        
        for _, row in jan_2023_payments.iterrows():
            trans_card_scheme = row['card_scheme']
            trans_is_credit = row['is_credit']
            trans_aci = row['aci']
            trans_ip_country = row['ip_country']
            trans_issuing_country = row['issuing_country']
            
            # Determine intracountry
            # Intracountry is true if ip_country == issuing_country
            trans_intracountry = (trans_ip_country == trans_issuing_country)
            
            # Check each fee
            for fee in fees:
                # Check card_scheme
                if fee['card_scheme'] != trans_card_scheme:
                    continue
                
                # Check account_type
                # If account_type list is empty, it matches any? 
                # Or does it need to be in the list? 
                # Looking at sample: "account_type": [] usually means no restriction or specific logic.
                # Let's assume if list is empty, it matches. If not empty, merchant_account_type must be in it.
                if fee['account_type'] and merchant_account_type not in fee['account_type']:
                    continue
                
                # Check merchant_category_code
                if merchant_mcc not in fee['merchant_category_code']:
                    continue
                
                # Check is_credit
                if fee['is_credit'] is not None and fee['is_credit'] != trans_is_credit:
                    continue
                
                # Check aci
                if trans_aci not in fee['aci']:
                    continue
                
                # Check intracountry
                if fee['intracountry'] is not None and fee['intracountry'] != trans_intracountry:
                    continue
                
                # If all conditions met, add ID
                applicable_fee_ids.add(fee['ID'])
        
        # Convert to sorted list for consistent output
        result = sorted(list(applicable_fee_ids))
        # Format as comma separated string
        answer_str = ", ".join(map(str, result))
        print(json.dumps({'answer': answer_str}))