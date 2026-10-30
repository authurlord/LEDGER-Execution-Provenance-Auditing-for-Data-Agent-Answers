import pandas as pd
import json
import os

# Load data
payments = pd.read_csv('payments.csv')
fees = json.load(open('fees.json'))
merchant_data = json.load(open('merchant_data.json'))
acquirer_countries = pd.read_csv('acquirer_countries.csv')

# Convert fees to DataFrame for easier handling
fees_df = pd.DataFrame(fees)

# Find Rafa_AI merchant data
rafa_ai_merchant = None
for m in merchant_data:
    if m['merchant'] == 'Rafa_AI':
        rafael_ai_merchant = m
        break

if rafael_ai_merchant is None:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

rafa_ai_mcc = rafael_ai_merchant['merchant_category_code']
rafa_ai_acquirers = rafael_ai_merchant['acquirer']

# Get acquirer countries for Rafa_AI
# Rafa_AI might have multiple acquirers. We need to know which acquirer processed which transaction?
# payments.csv does not have 'acquirer' column, only 'acquirer_country'.
# Wait, payments.csv has 'acquirer_country'.
# So we don't need to map acquirer name to country for the transaction itself, it's already there.
# But we need to know if the transaction is intracountry.
# Intracountry: acquirer_country == issuing_country.

# Filter payments for Rafa_AI in May 2023
# May 2023: Day of year 121 to 151
rafa_payments = payments[
    (payments['merchant'] == 'Rafa_AI') &
    (payments['year'] == 2023) &
    (payments['day_of_year'] >= 121) &
    (payments['day_of_year'] <= 151)
]

# Find fee ID=787
fee_787 = fees_df[fees_df['ID'] == 787]
if fee_787.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
    exit()

fee_787 = fee_787.iloc[0]
original_rate = fee_787['rate']
new_rate = 1
fixed_amount = fee_787['fixed_amount']
fee_card_scheme = fee_787['card_scheme']
fee_mcc_list = fee_787['merchant_category_code']
fee_is_credit = fee_787['is_credit']
fee_aci_list = fee_787['aci']
fee_intracountry = fee_787['intracountry']

# Function to check if a transaction matches the fee
def matches_fee(row, fee):
    # Check card scheme
    if row['card_scheme'] != fee['card_scheme']:
        return False
    
    # Check merchant category code
    # We need the MCC of the merchant. Since we are filtering for Rafa_AI, we know its MCC.
    if row['merchant'] == 'Rafa_AI':
        mcc = rafael_ai_mcc
    else:
        # This shouldn't happen as we filtered for Rafa_AI
        return False
        
    if mcc not in fee['merchant_category_code']:
        return False
    
    # Check is_credit
    if fee['is_credit'] is not None:
        if row['is_credit'] != fee['is_credit']:
            return False
    
    # Check ACI
    if row['aci'] not in fee['aci']:
        return False
    
    # Check intracountry
    if fee['intracountry'] is not None:
        # Determine if transaction is intracountry
        # Intracountry: acquirer_country == issuing_country
        is_intracountry = (row['acquirer_country'] == row['issuing_country'])
        
        # If fee intracountry is True/1, it applies to intracountry transactions
        # If fee intracountry is False/0, it applies to intercountry transactions
        # Note: The sample shows 0.0. Let's assume 0.0 means False.
        if fee['intracountry']:
            if not is_intracountry:
                return False
        else:
            if is_intracountry:
                return False
                
    return True

# Filter matching transactions
matching_payments = rafael_payments[rafael_payments.apply(lambda row: matches_fee(row, fee_787), axis=1)]

# Calculate delta
# Delta = New Fee - Old Fee
# New Fee = fixed + amount * (1 / 100)
# Old Fee = fixed + amount * (original_rate / 100)
# Delta = amount * (1 - original_rate) / 100

if matching_payments.empty:
    delta = 0.0
else:
    total_amount = matching_payments['eur_amount'].sum()
    delta = total_amount * (1 - original_rate) / 100

# Round to 14 decimals
delta_rounded = round(delta, 14)

print(json.dumps({'answer': delta_rounded}))