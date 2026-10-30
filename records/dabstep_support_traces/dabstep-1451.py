import pandas as pd
import json

# Load fees.json
with open('fees.json', 'r') as f:
    fees = json.load(f)

# Filter for GlobalCard and is_credit == True
global_card_credit_fees = [
    fee for fee in fees
    if fee.get('card_scheme') == 'GlobalCard' and fee.get('is_credit') == True
]

# If no records match, the answer might be 'Not Applicable'
if not global_card_credit_fees:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Calculate cost for each ACI
    aci_costs = {}
    for fee in global_card_credit_fees:
        fixed = fee.get('fixed_amount', 0)
        rate = fee.get('rate', 0)
        # Cost for 100 euros
        cost = fixed + (100 * rate / 100)
        
        acis = fee.get('aci', [])
        for aci in acis:
            if aci not in aci_costs:
                aci_costs[aci] = cost
            else:
                # If an ACI appears in multiple fee structures, we might need to consider the max cost?
                # The question asks for the "most expensive ACI". 
                # If an ACI is associated with multiple fee records, which cost do we use?
                # Typically, the merchant would choose the best fee, but the question asks for the most expensive.
                # However, usually, an ACI is a characteristic of the transaction.
                # If multiple fee structures apply, the actual fee charged would be the one that applies.
                # But here, we are looking for the ACI that *would be* the most expensive.
                # Let's assume we take the maximum cost associated with an ACI across all applicable fee records.
                aci_costs[aci] = max(aci_costs[aci], cost)
    
    if not aci_costs:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        # Find the maximum cost
        max_cost = max(aci_costs.values())
        
        # Get all ACIs with the maximum cost
        most_expensive_acis = [aci for aci, cost in aci_costs.items() if cost == max_cost]
        
        # Sort alphabetically and pick the first one
        most_expensive_acis.sort()
        answer = most_expensive_acis[0]
        
        print(json.dumps({'answer': [answer]}))