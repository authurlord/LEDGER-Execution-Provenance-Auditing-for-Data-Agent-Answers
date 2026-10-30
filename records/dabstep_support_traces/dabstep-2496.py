import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
with open('merchant_data.json', 'r') as f:
    merchants = json.load(f)
with open('fees.json', 'r') as f:
    fees = json.load(f)
acquirer_countries = pd.read_csv('acquirer_countries.csv')

# Filter for Golfclub_Baron_Friso in 2023
merchant_name = 'Golfclub_Baron_Friso'
merchant_payments = payments[(payments['merchant'] == merchant_name) & (payments['year'] == 2023)]

# Get merchant details
merchant_info = None
for m in merchants:
    if m['merchant'] == merchant_name:
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    merchant_mcc = merchant_info['merchant_category_code']
    
    # Find fee ID 709
    fee_709 = None
    for fee in fees:
        if fee['ID'] == 709:
            fee_709 = fee
            break
            
    if fee_709 is None:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        rate_old = fee_709['rate']
        rate_new = 99
        
        # Criteria for fee 709
        fee_card_scheme = fee_709['card_scheme']
        fee_is_credit = fee_709['is_credit']
        fee_mcc_list = fee_709['merchant_category_code']
        fee_aci_list = fee_709['aci']
        fee_intracountry = fee_709['intracountry']
        
        # Check if merchant MCC is in fee's MCC list
        if merchant_mcc not in fee_mcc_list:
            # No transactions match this fee
            print(json.dumps({'answer': 0.0}))
        else:
            # Filter payments that match the fee criteria
            # 1. Card Scheme
            mask = merchant_payments['card_scheme'] == fee_card_scheme
            
            # 2. Is Credit
            if fee_is_credit is not None:
                mask = mask & (merchant_payments['is_credit'] == fee_is_credit)
            # If fee_is_credit is None, it applies to both? Or is it a specific type?
            # Usually, if is_credit is None in fee, it might mean it applies to all, 
            # or it's not a filter. Let's assume if None, it's not a filter.
            
            # 3. ACI
            if fee_aci_list:
                mask = mask & (merchant_payments['aci'].isin(fee_aci_list))
                
            # 4. Intracountry
            if fee_intracountry is not None:
                # We need to determine if transaction is intracountry
                # Intracountry: issuing_country == acquirer_country
                # We have issuing_country and acquirer_country in payments.csv
                # So we just check if they are equal
                mask = mask & (merchant_payments['issuing_country'] == merchant_payments['acquirer_country'])
            
            matching_payments = merchant_payments[mask]
            total_amount = matching_payments['eur_amount'].sum()
            
            # Delta = (rate_new - rate_old) / 100 * total_amount
            delta = (rate_new - rate_old) / 100.0 * total_amount
            
            print(json.dumps({'answer': round(delta, 14)}))