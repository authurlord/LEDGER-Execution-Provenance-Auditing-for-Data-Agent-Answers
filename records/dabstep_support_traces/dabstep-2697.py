import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
with open('fees.json', 'r') as f:
    fees_data = json.load(f)
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)

# Filter for Belles_cookbook_store in January (day_of_year 1-31) and fraudulent
belles_fraud = payments[
    (payments['merchant'] == 'Belles_cookbook_store') &
    (payments['day_of_year'] <= 31) &
    (payments['has_fraudulent_dispute'] == True)
]

if belles_fraud.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Get merchant details
    merchant_info = merchant_data[merchant_data['merchant'] == 'Belles_cookbook_store']
    if merchant_info.empty:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        mcc = merchant_info.iloc[0]['merchant_category_code']
        
        # Determine the card scheme and is_credit status from the fraudulent transactions
        # It's possible there are multiple card schemes or credit types, but typically a merchant uses one.
        # Let's assume we need to handle each unique combination if there are multiple.
        # However, the question asks for "the preferred choice", implying a single answer.
        # Let's look at the unique card schemes in the fraudulent transactions.
        
        unique_combos = belles_fraud[['card_scheme', 'is_credit']].drop_duplicates()
        
        best_aci = None
        min_fee = float('inf')
        best_card_scheme = None
        
        for _, combo in unique_combos.iterrows():
            card_scheme = combo['card_scheme']
            is_credit = combo['is_credit']
            
            # Filter fees for this card scheme, is_credit, and merchant category code
            # The fees.json structure:
            # - card_scheme
            # - is_credit (can be True, False, or null meaning any)
            # - merchant_category_code (list)
            # - aci (list)
            # - fixed_amount
            # - rate
            
            matching_fees = []
            for fee in fees_data:
                if fee['card_scheme'] != card_scheme:
                    continue
                if fee['is_credit'] is not None and fee['is_credit'] != is_credit:
                    continue
                if mcc not in fee['merchant_category_code']:
                    continue
                matching_fees.append(fee)
            
            if not matching_fees:
                continue
                
            # For each matching fee, we can potentially use any ACI listed in that fee's 'aci' list
            # But wait, the fee is associated with specific ACIs. 
            # If we change the ACI, we must use a fee that supports that ACI.
            # So for each fee entry, the available ACIs are those in fee['aci'].
            # We need to calculate the fee for each ACI in fee['aci'].
            
            # However, different ACIs might have different fees even for the same card scheme/is_credit/mcc?
            # In the fees.json, the fee (fixed_amount, rate) is tied to a list of ACIs.
            # So all ACIs in a single fee entry have the same cost structure.
            
            # Let's collect all possible ACIs and their associated fee structures.
            aci_fees = {}
            for fee in matching_fees:
                for aci in fee['aci']:
                    if aci not in aci_fees:
                        aci_fees[aci] = {'fixed_amount': fee['fixed_amount'], 'rate': fee['rate']}
            
            # Now calculate the average fee for each ACI across all fraudulent transactions for this combo
            # Or just the total fee? The question asks for "lowest possible fees".
            # It implies we want to minimize the cost per transaction or total cost.
            # Since the fee structure is linear (fixed + rate * amount), minimizing the rate/fixed will minimize total cost.
            # Let's calculate the average eur_amount for this combo to estimate the fee per transaction.
            avg_amount = belles_fraud[belles_fraud['card_scheme'] == card_scheme]['eur_amount'].mean()
            
            for aci, fee_struct in aci_fees.items():
                fee_per_txn = fee_struct['fixed_amount'] + (avg_amount * fee_struct['rate'] / 100)
                if fee_per_txn < min_fee:
                    min_fee = fee_per_txn
                    best_aci = aci
                    best_card_scheme = card_scheme
        
        if best_aci is not None:
            # Round to 2 decimals
            print(json.dumps({'answer': f"{best_card_scheme}:{round(min_fee, 2)}"}))
        else:
            print(json.dumps({'answer': 'Not Applicable'}))