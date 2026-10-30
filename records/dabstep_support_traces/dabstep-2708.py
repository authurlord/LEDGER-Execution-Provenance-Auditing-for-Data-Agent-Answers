import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
with open('fees.json', 'r') as f:
    fees = json.load(f)
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)

# Filter for Crossfit_Hanna in March 2023 (day_of_year 60-90)
# Note: The sample data is for 2023. We'll assume the question refers to 2023.
# March 1, 2023 is day 60, March 31, 2023 is day 90.
crossfit_hanna_march_fraud = payments[
    (payments['merchant'] == 'Crossfit_Hanna') &
    (payments['year'] == 2023) &
    (payments['day_of_year'] >= 60) &
    (payments['day_of_year'] <= 90) &
    (payments['has_fraudulent_dispute'] == True)
]

# If there are no fraudulent transactions, the answer is Not Applicable
if crossfit_hanna_march_fraud.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Get the card schemes and current ACIs for these transactions
    card_schemes = crossfit_hanna_march_fraud['card_scheme'].unique()
    
    # For each card scheme, we need to find the best ACI
    # But the question asks for the preferred choice considering the lowest possible fees.
    # We need to calculate the total cost for each possible ACI for each card scheme.
    
    # First, let's get the merchant category code for Crossfit_Hanna
    merchant_info = merchant_data[merchant_data['merchant'] == 'Crossfit_Hanna']
    if merchant_info.empty:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        mcc = merchant_info.iloc[0]['merchant_category_code']
        
        # For each card scheme, find the best ACI
        best_aci = None
        min_cost = float('inf')
        
        for cs in card_schemes:
            # Get the transactions for this card scheme
            cs_transactions = crossfit_hanna_march_fraud[crossfit_hanna_march_fraud['card_scheme'] == cs]
            
            # Find all possible ACIs for this card scheme and MCC
            possible_acis = set()
            for fee_entry in fees:
                if fee_entry['card_scheme'] == cs and mcc in fee_entry['merchant_category_code']:
                    for aci in fee_entry['aci']:
                        possible_acis.add(aci)
            
            # If no possible ACIs, skip
            if not possible_acis:
                continue
            
            # For each possible ACI, calculate the total cost
            for aci in possible_acis:
                total_cost = 0
                for _, row in cs_transactions.iterrows():
                    # Find the fee entry for this card scheme, MCC, and ACI
                    fee_entry = None
                    for entry in fees:
                        if (entry['card_scheme'] == cs and 
                            mcc in entry['merchant_category_code'] and 
                            aci in entry['aci'] and
                            (entry['is_credit'] is None or entry['is_credit'] == row['is_credit'])):
                            fee_entry = entry
                            break
                    
                    if fee_entry:
                        fixed = fee_entry['fixed_amount']
                        rate = fee_entry['rate']
                        cost = fixed + row['eur_amount'] * rate / 100
                        total_cost += cost
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_aci = f"{cs}:{aci}"
        
        if best_aci:
            print(json.dumps({'answer': f"{best_aci}:{round(min_cost, 2)}"}))
        else:
            print(json.dumps({'answer': 'Not Applicable'}))