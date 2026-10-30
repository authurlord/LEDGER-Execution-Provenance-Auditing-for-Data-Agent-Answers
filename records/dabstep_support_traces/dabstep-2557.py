import pandas as pd
import json

# Load fees
with open('fees.json', 'r') as f:
    fees = json.load(f)

# Find Fee 64
fee_64 = None
for fee in fees:
    if fee['ID'] == 64:
        fee_64 = fee
        break

if fee_64 is None:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Load merchant data
    with open('merchant_data.json', 'r') as f:
        merchants = json.load(f)
    
    # Create a dataframe for merchants
    df_merchants = pd.DataFrame(merchants)
    
    # Analyze Fee 64 attributes
    # We need to know what the original account_type condition was.
    # If account_type is [], it usually means it applies to all account types.
    # If it is ['A', 'B'], it applies to those.
    
    original_account_types = fee_64.get('account_type', [])
    new_account_types = ['S']
    
    # Determine which merchants are affected.
    # A merchant is affected if the set of account types they possess intersects with the change in applicability.
    # Specifically:
    # 1. If the fee originally applied to their account type, but now does NOT (because they are not 'S').
    # 2. If the fee originally did NOT apply to their account type, but now DOES (because they ARE 'S').
    
    # Let's identify the account types of each merchant.
    # Note: A merchant might have multiple account types? The schema shows "account_type": "F" (string).
    # So each merchant has a single account type.
    
    affected_merchants = []
    
    for merchant in merchants:
        m_name = merchant['merchant']
        m_account_type = merchant['account_type']
        
        # Check if original fee applied to this merchant's account type
        # If original_account_types is empty, it applies to all.
        if len(original_account_types) == 0:
            original_applies = True
        else:
            original_applies = m_account_type in original_account_types
            
        # Check if new fee applies to this merchant's account type
        if len(new_account_types) == 0:
            new_applies = True
        else:
            new_applies = m_account_type in new_account_types
            
        # If the applicability changes, the merchant is affected
        if original_applies != new_applies:
            affected_merchants.append(m_name)
            
    # Sort the list for consistent output
    affected_merchants.sort()
    
    # Format the answer
    if not affected_merchants:
        answer = ""
    else:
        answer = ", ".join(affected_merchants)
        
    print(json.dumps({'answer': answer}))