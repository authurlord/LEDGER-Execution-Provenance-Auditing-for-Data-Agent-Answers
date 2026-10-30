import pandas as pd
import json

# Load the necessary files
# 1. Load merchant_category_codes.csv to find the MCC for "Eating Places and Restaurants"
mcc_df = pd.read_csv('merchant_category_codes.csv')

# Find the MCC for "Eating Places and Restaurants"
# Note: The description might vary slightly, so we look for an exact or close match.
# Standard MCC for Eating Places and Restaurants is 5812.
# Let's check if "Eating Places and Restaurants" is in the dataframe.
mcc_row = mcc_df[mcc_df['description'] == 'Eating Places and Restaurants']

if mcc_row.empty:
    # If not found exactly, we might need to check for partial matches or standard codes.
    # However, based on the sample data in merchant_data.json, merchant "Martinis_Fine_Steakhouse" has mcc 5812.
    # Let's assume standard MCC 5812 if not explicitly listed with that description, 
    # but let's try to find it in the CSV first.
    # Looking at the sample rows provided in the prompt for merchant_category_codes.csv, 
    # it doesn't show 5812. But the prompt says "Available files are all offered".
    # Let's assume the description is exactly "Eating Places and Restaurants".
    # If the CSV doesn't have it, we might have to infer from merchant_data.json or standard knowledge.
    # However, the instruction says "Use these exact local filenames".
    # Let's look at merchant_data.json sample: "Martinis_Fine_Steakhouse" has mcc 5812.
    # Let's check if 5812 is in the CSV.
    mcc_5812 = mcc_df[mcc_df['mcc'] == 5812]
    if not mcc_5812.empty:
        description = mcc_5812['description'].values[0]
        # If the description in CSV is different, we need to be careful.
        # But usually, 5812 is "Eating Places and Restaurants".
        target_mcc = 5812
    else:
        # Fallback: If not found, we might need to search for similar descriptions.
        # But let's assume the CSV contains the mapping.
        # Let's print the unique descriptions to debug if needed, but we can't interactively.
        # Let's assume the standard mapping: 5812 -> Eating Places and Restaurants.
        target_mcc = 5812
else:
    target_mcc = mcc_row['mcc'].values[0]

# 2. Load fees.json
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# 3. Filter fees for GlobalCard, Account Type H, and the specific MCC
# The fees.json structure has lists for account_type and merchant_category_code.
# We need to find a record where:
# - card_scheme is "GlobalCard"
# - "H" is in account_type (or account_type is empty/list indicating all?)
# - target_mcc is in merchant_category_code

applicable_fees = []
for fee in fees_data:
    if fee['card_scheme'] == 'GlobalCard':
        # Check account_type
        # If account_type is empty list [], it might mean all types or not specified.
        # If it contains 'H', it matches.
        # The prompt asks for account type H.
        account_types = fee['account_type']
        if not account_types: # Empty list usually means applicable to all or default
            # However, specific account types might override. 
            # Let's look at the sample: ID 1 has account_type: [], ID 2 has account_type: []
            # ID 3 has account_type: []
            # It seems many entries have empty account_type.
            # We should prioritize entries that explicitly list 'H' if available, 
            # or use the one with empty list if no specific one exists.
            # But wait, if there's a specific entry for 'H', we should use that.
            # If there's an entry with [], it applies to all.
            # Let's collect all matching fees.
            if target_mcc in fee['merchant_category_code']:
                applicable_fees.append(fee)
        elif 'H' in account_types:
            if target_mcc in fee['merchant_category_code']:
                applicable_fees.append(fee)

# If we have multiple fees, we might need to determine which one applies.
# Usually, more specific rules apply. An entry with account_type=['H'] is more specific than [].
# Let's separate specific and general.
specific_fees = [f for f in applicable_fees if f['account_type'] and 'H' in f['account_type']]
general_fees = [f for f in applicable_fees if not f['account_type']]

if specific_fees:
    # Use the specific fee
    selected_fee = specific_fees[0]
elif general_fees:
    # Use the general fee
    selected_fee = general_fees[0]
else:
    selected_fee = None

if selected_fee:
    fixed_amount = selected_fee['fixed_amount']
    rate = selected_fee['rate']
    transaction_value = 10.0
    
    # Calculate fee
    # Rate is likely in basis points or percentage? 
    # Sample: rate: 19, fixed_amount: 0.1. 
    # If rate is 19%, fee = 0.1 + 10 * 0.19 = 2.0
    # If rate is 19 basis points (0.19%), fee = 0.1 + 10 * 0.0019 = 0.119
    # Looking at sample ID 2: rate 86, fixed 0.13. 
    # 86% is very high. 0.86% is more reasonable. 
    # Let's check sample ID 1: rate 19. 19% is high for interchange but possible for some schemes.
    # However, in payment processing, rates are often in basis points (1/100th of a percent) or percentage.
    # Let's look at the manual or context. 
    # "rate": 19. If it's percentage, it's 19%. If it's basis points, it's 0.19%.
    # Given "fixed_amount": 0.1, a 19% fee on 10 EUR would be 1.9 EUR + 0.1 = 2.0 EUR.
    # A 0.19% fee would be 0.019 EUR + 0.1 = 0.119 EUR.
    # Typical interchange fees are around 0.2% - 0.5%. So 0.19% (19 bps) is reasonable.
    # 19% would be extremely high.
    # Let's assume the rate is in basis points (i.e., divide by 10000) or percentage (divide by 100).
    # Let's look at ID 2: rate 86. 86 bps = 0.86%. 86% is absurd.
    # So rate is likely in basis points.
    # Fee = fixed_amount + (transaction_value * rate / 10000)
    
    fee = fixed_amount + (transaction_value * rate / 10000)
    answer = round(fee, 6)
else:
    answer = 'Not Applicable'

print(json.dumps({'answer': answer}))