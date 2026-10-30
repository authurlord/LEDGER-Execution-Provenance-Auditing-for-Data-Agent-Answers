import pandas as pd
import json
import os

# Load fees.json
with open('fees.json', 'r') as f:
    fees = json.load(f)

# Find fee ID 398
fee_398 = None
for fee in fees:
    if fee['ID'] == 398:
        fee_398 = fee
        break

if fee_398 is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    original_rate = fee_398['rate']
    fee_card_scheme = fee_398['card_scheme']
    fee_mcc_list = fee_398['merchant_category_code']
    fee_is_credit = fee_398['is_credit']
    fee_aci_list = fee_398['aci']

    # Load merchant_data.json
    with open('merchant_data.json', 'r') as f:
        merchants = json.load(f)

    # Find Belles_cookbook_store
    merchant_info = None
    for m in merchants:
        if m['merchant'] == 'Belles_cookbook_store':
            merchant_info = m
            break

    if merchant_info is None:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        merchant_mcc = merchant_info['merchant_category_code']
        
        # Check if merchant MCC is in fee's MCC list
        if merchant_mcc not in fee_mcc_list:
            # If the merchant's MCC is not in the fee's MCC list, no transactions match
            # However, we should still check if there are any transactions that might match
            # based on other criteria? Usually, MCC is a hard filter.
            # Let's assume if MCC doesn't match, delta is 0.
            sum_amount = 0
        else:
            # Load payments.csv
            payments = pd.read_csv('payments.csv')
            
            # Filter for Belles_cookbook_store in October 2023
            # October 2023: Day of year 274 to 305 (2023 is not a leap year)
            # Jan: 31, Feb: 28, Mar: 31, Apr: 30, May: 31, Jun: 30, Jul: 31, Aug: 31, Sep: 30
            # Sum: 31+28+31+30+31+30+31+31+30 = 273. So Oct 1 is day 274.
            # Oct has 31 days. So Oct 31 is day 274 + 30 = 304? 
            # Let's re-calculate:
            # Jan 31, Feb 28 (59), Mar 31 (90), Apr 30 (120), May 31 (151), Jun 30 (181), Jul 31 (212), Aug 31 (243), Sep 30 (273).
            # Oct 1 is day 274. Oct 31 is day 274 + 30 = 304.
            # So day_of_year range is [274, 304].
            
            oct_2023_mask = (payments['merchant'] == 'Belles_cookbook_store') & \
                            (payments['year'] == 2023) & \
                            (payments['day_of_year'] >= 274) & \
                            (payments['day_of_year'] <= 304)
            
            oct_payments = payments[oct_2023_mask]
            
            # Filter by fee criteria
            # 1. card_scheme
            mask = oct_payments['card_scheme'] == fee_card_scheme
            
            # 2. is_credit
            if fee_is_credit is not None:
                mask = mask & (oct_payments['is_credit'] == fee_is_credit)
            
            # 3. aci
            if fee_aci_list:
                mask = mask & (oct_payments['aci'].isin(fee_aci_list))
            
            # 4. merchant_category_code is already checked globally, but let's be safe
            # The fee applies to the merchant's MCC. Since we already checked merchant_mcc in fee_mcc_list,
            # all transactions from this merchant are eligible regarding MCC.
            
            matching_payments = oct_payments[mask]
            sum_amount = matching_payments['eur_amount'].sum()
        
        # Calculate delta
        # Delta = (New Rate - Old Rate) / 100 * Sum Amount
        delta = (99 - original_rate) / 100 * sum_amount
        
        # Round to 14 decimals
        answer = round(delta, 14)
        print(json.dumps({'answer': answer}))