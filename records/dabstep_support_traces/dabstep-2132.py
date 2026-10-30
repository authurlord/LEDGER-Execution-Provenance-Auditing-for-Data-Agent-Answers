import pandas as pd
import json

# Load data
with open('fees.json', 'r') as f:
    fees = json.load(f)

with open('merchant_data.json', 'r') as f:
    merchants = json.load(f)

payments = pd.read_csv('payments.csv')

# Create a lookup for merchant data
merchant_lookup = {m['merchant']: m for m in merchants}

# Find fee ID 65
fee_65 = None
for fee in fees:
    if fee['ID'] == 65:
        fee_65 = fee
        break

if fee_65 is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Filter transactions for Golfclub_Baron_Friso in March 2023
    # March 2023: 2023 is not a leap year.
    # Jan: 31, Feb: 28. March 1 is day 60. March 31 is day 90.
    merchant_name = "Golfclub_Baron_Friso"
    
    # Get merchant info
    if merchant_name not in merchant_lookup:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        merchant_info = merchant_lookup[merchant_name]
        merchant_mcc = merchant_info['merchant_category_code']
        
        # Filter payments
        # Note: The problem asks for March 2023.
        # We filter by year=2023 and day_of_year between 60 and 90 inclusive.
        mask = (
            (payments['merchant'] == merchant_name) &
            (payments['year'] == 2023) &
            (payments['day_of_year'] >= 60) &
            (payments['day_of_year'] <= 90)
        )
        relevant_payments = payments[mask]
        
        if relevant_payments.empty:
            print(json.dumps({'answer': 'Not Applicable'}))
        else:
            # Check which transactions match fee ID 65
            # Fee 65 criteria:
            # card_scheme == fee['card_scheme']
            # merchant_category_code in fee['merchant_category_code']
            # is_credit matches fee['is_credit']
            # aci in fee['aci']
            
            fee_65_card_scheme = fee_65['card_scheme']
            fee_65_mccs = fee_65['merchant_category_code']
            fee_65_is_credit = fee_65['is_credit']
            fee_65_aci = fee_65['aci']
            fee_65_rate = fee_65['rate']
            
            delta_sum = 0.0
            
            for _, row in relevant_payments.iterrows():
                # Check card scheme
                if row['card_scheme'] != fee_65_card_scheme:
                    continue
                
                # Check MCC
                if merchant_mcc not in fee_65_mccs:
                    continue
                
                # Check is_credit
                if fee_65_is_credit is not None:
                    if row['is_credit'] != fee_65_is_credit:
                        continue
                # If fee_65_is_credit is None, it applies to both, so no check needed
                
                # Check ACI
                if row['aci'] not in fee_65_aci:
                    continue
                
                # This transaction matches fee ID 65
                # Delta = eur_amount * (new_rate - old_rate) / 100
                delta = row['eur_amount'] * (99 - fee_65_rate) / 100
                delta_sum += delta
            
            # Round to 14 decimals
            answer = round(delta_sum, 14)
            print(json.dumps({'answer': answer}))