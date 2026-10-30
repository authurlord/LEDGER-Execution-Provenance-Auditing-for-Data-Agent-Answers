import pandas as pd
import json

# Load fees.json
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# Filter for TransactPlus and is_credit == True
transactplus_credit_fees = [
    item for item in fees_data 
    if item.get('card_scheme') == 'TransactPlus' and item.get('is_credit') == True
]

# If no such records, answer is Not Applicable
if not transactplus_credit_fees:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Collect all ACIs from these records
    all_acis = set()
    for item in transactplus_credit_fees:
        for aci in item.get('aci', []):
            all_acis.add(aci)
    
    # Calculate fee for each ACI
    # Note: An ACI might appear in multiple fee records with different rates/fixed amounts.
    # The question asks for the "most expensive" ACI. This implies we should look at the 
    # worst-case (highest) fee associated with that ACI across all applicable fee structures?
    # Or does it mean for a specific transaction, which ACI yields the highest fee?
    # Usually, ACI determines the fee tier. If an ACI appears in multiple tiers, 
    # we should probably consider the maximum possible fee for that ACI to be "most expensive".
    # However, typically an ACI maps to a specific fee structure. 
    # Let's assume we calculate the fee for each (ACI, FeeRecord) pair and find the max.
    
    aci_max_fees = {}
    
    for item in transactplus_credit_fees:
        fixed = item.get('fixed_amount', 0)
        if fixed is None:
            fixed = 0
        rate = item.get('rate', 0)
        if rate is None:
            rate = 0
        
        amount = 10000
        fee = fixed + (amount * rate / 100)
        
        for aci in item.get('aci', []):
            if aci not in aci_max_fees:
                aci_max_fees[aci] = fee
            else:
                if fee > aci_max_fees[aci]:
                    aci_max_fees[aci] = fee
    
    if not aci_max_fees:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        max_fee = max(aci_max_fees.values())
        most_expensive_acis = [aci for aci, fee in aci_max_fees.items() if fee == max_fee]
        most_expensive_acis.sort()
        answer = most_expensive_acis[0]
        print(json.dumps({'answer': [answer]}))