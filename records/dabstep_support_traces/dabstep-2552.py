import pandas as pd
import json
import os

# Load data
with open('merchant_data.json', 'r') as f:
    merchants = json.load(f)

with open('fees.json', 'r') as f:
    fees = json.load(f)

payments = pd.read_csv('payments.csv')

# Find Rafa_AI's data
rafa_ai_data = None
for m in merchants:
    if m['merchant'] == 'Rafa_AI':
        rafael_ai_data = m
        break

if rafael_ai_data is None:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

actual_mcc = rafael_ai_data['merchant_category_code']
account_type = rafael_ai_data['account_type']

# Filter 2023 payments for Rafa_AI
rafa_payments = payments[(payments['merchant'] == 'Rafa_AI') & (payments['year'] == 2023)]

def calculate_fees(payments_df, mcc_to_use):
    total_fee = 0.0
    for _, row in payments_df.iterrows():
        card_scheme = row['card_scheme']
        is_credit = row['is_credit']
        aci = row['aci']
        eur_amount = row['eur_amount']
        issuing_country = row['issuing_country']
        acquirer_country = row['acquirer_country']
        
        # Find matching fee rules
        matching_rules = []
        for fee_rule in fees:
            # Check card_scheme
            if fee_rule['card_scheme'] != card_scheme:
                continue
            
            # Check merchant_category_code
            if mcc_to_use not in fee_rule['merchant_category_code']:
                continue
            
            # Check is_credit
            if fee_rule['is_credit'] is not None and fee_rule['is_credit'] != is_credit:
                continue
            
            # Check aci
            if fee_rule['aci'] and aci not in fee_rule['aci']:
                continue
            
            # Check intracountry
            intracountry_val = fee_rule['intracountry']
            is_intracountry = (issuing_country == acquirer_country)
            
            if intracountry_val is not None:
                # If intracountry is specified, it must match
                # Assuming intracountry_val being not None implies it's a flag for intracountry rules
                # The value 0.0 might just be a placeholder. Let's assume if it's not None, it applies to intracountry.
                if not is_intracountry:
                    continue
            
            matching_rules.append(fee_rule)
        
        # Select the best rule
        # Priority: Rules with specific intracountry match > Rules with intracountry None
        # If multiple, pick the first one (or lowest ID)
        best_rule = None
        
        # Separate rules into intracountry-specific and general
        intracountry_specific = [r for r in matching_rules if r['intracountry'] is not None]
        general = [r for r in matching_rules if r['intracountry'] is None]
        
        if is_intracountry:
            if intracountry_specific:
                best_rule = min(intracountry_specific, key=lambda x: x['ID'])
            elif general:
                best_rule = min(general, key=lambda x: x['ID'])
        else:
            # Not intracountry
            # If there are intracountry-specific rules, they don't apply
            # So we only look at general rules
            if general:
                best_rule = min(general, key=lambda x: x['ID'])
            # If no general rule, and no intracountry rule applies, fee is 0?
        
        if best_rule:
            fixed = best_rule['fixed_amount'] if best_rule['fixed_amount'] is not None else 0
            rate = best_rule['rate'] if best_rule['rate'] is not None else 0
            fee = fixed + (eur_amount * rate / 100)
            total_fee += fee
            
    return total_fee

actual_fees = calculate_fees(rafa_payments, actual_mcc)
hypothetical_fees = calculate_fees(rafa_payments, 7523)

delta = hypothetical_fees - actual_fees

print(json.dumps({'answer': round(delta, 6)}))