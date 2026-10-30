import pandas as pd
import json
import os

# Load payments data
payments = pd.read_csv('payments.csv')

# Load merchant data
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)

# Load fees data
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# 1. Get merchant info for Martinis_Fine_Steakhouse
merchant_name = "Martinis_Fine_Steakhouse"
merchant_info = None
for m in merchant_data:
    if m['merchant'] == merchant_name:
        merchant_info = m
        break

if merchant_info is None:
    # If merchant not found, answer is Not Applicable or No? 
    # The question implies the merchant exists.
    print(json.dumps({'answer': 'no'}))
else:
    mcc = merchant_info['merchant_category_code']
    
    # 2. Filter payments for this merchant
    merchant_payments = payments[payments['merchant'] == merchant_name]
    
    if len(merchant_payments) == 0:
        # No transactions, fraud rate is 0%
        fraud_rate = 0.0
    else:
        total_transactions = len(merchant_payments)
        fraudulent_disputes = merchant_payments['has_fraudulent_dispute'].sum()
        fraud_rate = (fraudulent_disputes / total_transactions) * 100

    # 3. Find applicable fraud thresholds from fees.json
    # We look for fee entries that include the merchant's MCC in their merchant_category_code list
    # and have a monthly_fraud_level defined.
    thresholds = []
    for fee in fees_data:
        if mcc in fee.get('merchant_category_code', []):
            mfl = fee.get('monthly_fraud_level')
            if mfl is not None:
                # Parse the threshold string, e.g., ">8.3%" -> 8.3
                if isinstance(mfl, str) and mfl.startswith('>'):
                    try:
                        threshold_val = float(mfl.replace('>', '').replace('%', ''))
                        thresholds.append(threshold_val)
                    except ValueError:
                        pass
    
    # 4. Determine if in danger
    # If the fraud rate exceeds ANY of the thresholds, it's in danger.
    in_danger = False
    if thresholds:
        for threshold in thresholds:
            if fraud_rate > threshold:
                in_danger = True
                break
    else:
        # If no specific fraud level thresholds are defined for this MCC in the fee structure,
        # we might assume no fine based on fraud rate alone, or check other conditions.
        # However, typically "high-fraud rate fine" implies a specific threshold.
        # If no threshold is found, we can't say it's in danger based on this metric.
        in_danger = False

    answer = 'yes' if in_danger else 'no'
    print(json.dumps({'answer': answer}))