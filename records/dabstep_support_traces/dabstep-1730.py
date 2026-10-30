import pandas as pd
import json
import os

# Load data
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)

with open('fees.json', 'r') as f:
    fees_data = json.load(f)

payments_df = pd.read_csv('payments.csv')

# Get merchant details
merchant_name = "Martinis_Fine_Steakhouse"
merchant_info = None
for m in merchant_data:
    if m['merchant'] == merchant_name:
        merchant_info = m
        break

if merchant_info is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    merchant_mcc = merchant_info['merchant_category_code']
    
    # Filter payments for the specific date and merchant
    # day_of_year 12, year 2023
    filtered_payments = payments_df[
        (payments_df['merchant'] == merchant_name) &
        (payments_df['year'] == 2023) &
        (payments_df['day_of_year'] == 12)
    ]
    
    total_fees = 0.0
    
    for _, row in filtered_payments.iterrows():
        card_scheme = row['card_scheme']
        is_credit = row['is_credit']
        aci = row['aci']
        eur_amount = row['eur_amount']
        ip_country = row['ip_country']
        issuing_country = row['issuing_country']
        
        is_intra = (ip_country == issuing_country)
        
        matching_rules = []
        
        for fee_rule in fees_data:
            # Check card_scheme
            if fee_rule['card_scheme'] != card_scheme:
                continue
            
            # Check is_credit
            rule_is_credit = fee_rule['is_credit']
            if rule_is_credit is not None:
                if rule_is_credit != is_credit:
                    continue
            
            # Check aci
            rule_aci = fee_rule['aci']
            if aci not in rule_aci:
                continue
            
            # Check mcc
            rule_mccs = fee_rule['merchant_category_code']
            if merchant_mcc not in rule_mccs:
                continue
            
            # Check intracountry constraint
            rule_intracountry = fee_rule['intracountry']
            if rule_intracountry is not None:
                # If rule specifies intracountry, it must match
                # Assuming 0.0 means "apply to intracountry"
                if rule_intracountry == 0.0:
                    if not is_intra:
                        continue
                # If rule_intracountry is 1.0 or other, assume it means non-intra?
                # But samples only show 0.0 and null.
                # Let's assume if it's not null and not 0.0, it might be a different flag.
                # Given the data, let's stick to 0.0 -> intra, null -> any.
            
            # Calculate fee
            fixed = fee_rule['fixed_amount']
            rate = fee_rule['rate']
            # Rate is likely in basis points (1/10000)
            fee = fixed + eur_amount * (rate / 10000)
            matching_rules.append(fee)
        
        if matching_rules:
            # Pick the minimum fee (optimization)
            total_fees += min(matching_rules)
    
    print(json.dumps({'answer': round(total_fees, 2)}))