import pandas as pd
import json

# Load fees
with open('fees.json', 'r') as f:
    fees = json.load(f)

# Find fee with ID 384
fee_384 = None
for fee in fees:
    if fee['ID'] == 384:
        fee_384 = fee
        break

if fee_384 is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Load payments
    payments = pd.read_csv('payments.csv')
    
    # Filter for Belles_cookbook_store in January 2023
    # January 2023: year=2023, day_of_year <= 31
    jan_2023_payments = payments[
        (payments['merchant'] == 'Belles_cookbook_store') &
        (payments['year'] == 2023) &
        (payments['day_of_year'] <= 31)
    ]
    
    if jan_2023_payments.empty:
        print(json.dumps({'answer': 0.0}))
    else:
        # Filter payments that match fee_384 criteria
        # Criteria:
        # card_scheme in fee_384['card_scheme']
        # merchant_category_code in fee_384['merchant_category_code']
        # is_credit == fee_384['is_credit'] (if not None)
        # aci in fee_384['aci']
        
        # Get merchant MCC
        with open('merchant_data.json', 'r') as f:
            merchants = json.load(f)
        
        merchant_info = None
        for m in merchants:
            if m['merchant'] == 'Belles_cookbook_store':
                merchant_info = m
                break
        
        merchant_mcc = merchant_info['merchant_category_code']
        
        # Check if merchant MCC is in the fee's MCC list
        if merchant_mcc not in fee_384['merchant_category_code']:
            # If the merchant's MCC is not covered by this fee, the fee doesn't apply?
            # Or does it apply to a subset? The fee has a list of MCCs.
            # If the merchant's MCC is not in the list, the fee rule doesn't match.
            print(json.dumps({'answer': 0.0}))
        else:
            # Filter payments
            # card_scheme
            valid_card_schemes = fee_384['card_scheme']
            # is_credit
            target_is_credit = fee_384['is_credit']
            # aci
            valid_acis = fee_384['aci']
            
            # Create mask
            mask = jan_2023_payments['card_scheme'].isin(valid_card_schemes)
            
            if target_is_credit is not None:
                mask = mask & (jan_2023_payments['is_credit'] == target_is_credit)
            
            mask = mask & jan_2023_payments['aci'].isin(valid_acis)
            
            matching_payments = jan_2023_payments[mask]
            
            if matching_payments.empty:
                print(json.dumps({'answer': 0.0}))
            else:
                old_rate = fee_384['rate']
                new_rate = 1
                
                # Delta = sum(eur_amount * (new_rate - old_rate) / 10000)
                # Assuming rate is in basis points
                delta = matching_payments['eur_amount'].sum() * (new_rate - old_rate) / 10000
                
                print(json.dumps({'answer': round(delta, 14)}))