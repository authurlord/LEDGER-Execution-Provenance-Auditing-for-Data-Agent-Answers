import pandas as pd
import json
import os

# Load data
payments = pd.read_csv('payments.csv')
merchant_data = json.load(open('merchant_data.json'))
fees_data = json.load(open('fees.json'))
acquirer_countries = pd.read_csv('acquirer_countries.csv')

# 1. Get merchant info
merchant_name = 'Belles_cookbook_store'
merchant_info = next((m for m in merchant_data if m['merchant'] == merchant_name), None)
if not merchant_info:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

mcc = merchant_info['merchant_category_code']
acquirer = merchant_info['acquirer'][0] # Assuming single acquirer or primary

# 2. Get acquirer country
acquirer_row = acquirer_countries[acquirer_countries['acquirer'] == acquirer]
if acquirer_row.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()
acquirer_country = acquirer_row.iloc[0]['country_code']

# 3. Filter January payments for the merchant
# January is day_of_year 1-31
jan_payments = payments[
    (payments['merchant'] == merchant_name) & 
    (payments['day_of_year'] >= 1) & 
    (payments['day_of_year'] <= 31)
]

if jan_payments.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

# 4. Define fee calculation function
def calculate_fee(transaction, fee_rule):
    # Check card_scheme
    if fee_rule['card_scheme'] != transaction['card_scheme']:
        return None
    
    # Check merchant_category_code
    if mcc not in fee_rule['merchant_category_code']:
        return None
    
    # Check is_credit
    rule_is_credit = fee_rule['is_credit']
    trans_is_credit = transaction['is_credit']
    if rule_is_credit is not None and rule_is_credit != trans_is_credit:
        return None
    
    # Check aci
    rule_aci = fee_rule['aci']
    trans_aci = transaction['aci']
    if rule_aci and trans_aci not in rule_aci:
        return None
    
    # Check intracountry
    # Determine if transaction is intracountry
    is_intracountry = transaction['issuing_country'] == acquirer_country
    
    rule_intracountry = fee_rule['intracountry']
    
    # If rule has intracountry specified, it applies to intracountry transactions
    # If rule has intracountry null, it applies to intercountry transactions?
    # Let's assume:
    # - If rule_intracountry is not None: applies if is_intracountry is True
    # - If rule_intracountry is None: applies if is_intracountry is False
    if rule_intracountry is not None:
        if not is_intracountry:
            return None
    else:
        if is_intracountry:
            return None
            
    # Calculate fee
    fixed = fee_rule['fixed_amount']
    rate = fee_rule['rate']
    amount = transaction['eur_amount']
    
    # Rate is in basis points (1/10000)
    fee = fixed + amount * (rate / 10000)
    return fee

# 5. Calculate fees for each transaction and aggregate by card_scheme
scheme_fees = {}

for _, trans in jan_payments.iterrows():
    best_fee = None
    for fee_rule in fees_data:
        fee = calculate_fee(trans, fee_rule)
        if fee is not None:
            if best_fee is None or fee < best_fee:
                best_fee = fee
    
    if best_fee is not None:
        scheme = trans['card_scheme']
        if scheme not in scheme_fees:
            scheme_fees[scheme] = []
        scheme_fees[scheme].append(best_fee)

# 6. Find scheme with minimum average fee
if not scheme_fees:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

min_avg_fee = float('inf')
best_scheme = None

for scheme, fees in scheme_fees.items():
    avg_fee = sum(fees) / len(fees)
    if avg_fee < min_avg_fee:
        min_avg_fee = avg_fee
        best_scheme = scheme

if best_scheme is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    answer = f"{best_scheme}:{min_avg_fee:.2f}"
    print(json.dumps({'answer': answer}))