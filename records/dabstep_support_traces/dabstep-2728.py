import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
with open('fees.json', 'r') as f:
    fees_data = json.load(f)
fees_df = pd.DataFrame(fees_data)
with open('merchant_data.json', 'r') as f:
    merchant_data = json.load(f)
merchant_df = pd.DataFrame(merchant_data)

# Filter for Crossfit_Hanna
crossfit_payments = payments[payments['merchant'] == 'Crossfit_Hanna']

# Filter for July (Day of Year 182-213 for non-leap, 183-214 for leap? 2023 is not leap. 2024 is leap. Let's assume standard July: 182-212 for non-leap, 183-213 for leap. Let's just use 182-213 to be safe for both, or check years.)
# 2023: Jan 1 is 1. July 1 is 182. July 31 is 212.
# 2024: Jan 1 is 1. July 1 is 183. July 31 is 213.
# Let's filter for day_of_year between 182 and 213 inclusive.
july_mask = (crossfit_payments['day_of_year'] >= 182) & (crossfit_payments['day_of_year'] <= 213)
crossfit_july = crossfit_payments[july_mask]

# Filter for fraudulent transactions
fraudulent = crossfit_july[crossfit_payments['has_fraudulent_dispute'] == True]

# Get merchant MCC
crossfit_merchant_info = merchant_df[merchant_df['merchant'] == 'Crossfit_Hanna']
if not crossfit_merchant_info.empty:
    mcc = crossfit_merchant_info.iloc[0]['merchant_category_code']
else:
    mcc = None

# If no fraudulent transactions in July, return Not Applicable
if fraudulent.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    # Identify current ACIs and their counts/amounts
    # We need to find the best ACI for each transaction type (card_scheme, is_credit)
    # Then aggregate to find the single best ACI for all fraudulent transactions?
    # Or is the question asking for the best ACI for the specific card scheme used?
    # The output format is {card_scheme}:{fee}. This suggests the answer is tied to a card scheme.
    # If there are multiple card schemes, we might need to pick the one with the highest volume or just report the best one?
    # Let's assume we need to find the ACI that minimizes the total fee for all fraudulent transactions.
    # But the output format includes card_scheme. This implies the answer is specific to a card scheme.
    # Maybe all fraudulent transactions are on one card scheme?
    
    # Let's group by card_scheme and is_credit
    groups = fraudulent.groupby(['card_scheme', 'is_credit'])
    
    # For each group, find the best ACI
    # We need to match fees based on card_scheme, mcc, is_credit, and aci
    
    # Prepare fee lookup
    # fees_df columns: ID, card_scheme, account_type, capture_delay, monthly_fraud_level, monthly_volume, merchant_category_code, is_credit, aci, fixed_amount, rate, intracountry
    
    # Function to calculate fee for a transaction given an ACI
    def get_fee_for_aci(row, aci_val, fees_df, mcc):
        # Filter fees by card_scheme
        scheme_fees = fees_df[fees_df['card_scheme'] == row['card_scheme']]
        # Filter by mcc (list in fees_df)
        mcc_fees = scheme_fees[scheme_fees['merchant_category_code'].apply(lambda x: mcc in x)]
        # Filter by is_credit
        # is_credit in fees_df can be None (any), True, or False
        if pd.isna(row['is_credit']):
            credit_fees = mcc_fees
        else:
            credit_fees = mcc_fees[mcc_fees['is_credit'].isin([row['is_credit'], None])]
        
        # Filter by aci
        aci_fees = credit_fees[credit_fees['aci'].apply(lambda x: aci_val in x)]
        
        if aci_fees.empty:
            return None # No fee structure found
        
        # If multiple, take the first or average? Usually specific rules apply.
        # Let's take the first one.
        fee_rule = aci_fees.iloc[0]
        fixed = fee_rule['fixed_amount']
        rate = fee_rule['rate']
        # Rate is likely in basis points (0.01%)
        # Fee = fixed + amount * (rate / 10000)
        fee = fixed + row['eur_amount'] * (rate / 10000)
        return fee

    # Calculate total fee for each ACI for each group
    # We want to find the ACI that minimizes the total fee across all fraudulent transactions.
    # But the output is {card_scheme}:{fee}. This suggests we report the best ACI for the card scheme with the most fraud?
    # Or maybe there's only one card scheme.
    
    # Let's collect all unique card schemes in fraudulent transactions
    unique_schemes = fraudulent['card_scheme'].unique()
    
    # If multiple schemes, we might need to pick the one with the highest total amount or count.
    # Let's assume we pick the scheme with the highest total fraudulent amount.
    
    scheme_totals = fraudulent.groupby('card_scheme')['eur_amount'].sum()
    best_scheme = scheme_totals.idxmax()
    
    # Filter fraudulent transactions for this scheme
    scheme_fraud = fraudulent[fraudulent['card_scheme'] == best_scheme]
    
    # Now, for this scheme, find the best ACI.
    # We need to consider all possible ACIs that appear in the fee structure for this scheme and MCC.
    # Possible ACIs: A, B, C, D, E, F (from manual/schema)
    
    # Get all possible ACIs from fees_df for this scheme and MCC
    scheme_fees = fees_df[fees_df['card_scheme'] == best_scheme]
    mcc_fees = scheme_fees[scheme_fees['merchant_category_code'].apply(lambda x: mcc in x)]
    
    # Collect all unique ACIs from these fee rules
    all_acis = set()
    for aci_list in mcc_fees['aci']:
        if isinstance(aci_list, list):
            all_acis.update(aci_list)
        else:
            all_acis.add(aci_list)
            
    # Calculate total fee for each ACI
    aci_total_fees = {}
    for aci in all_acis:
        total_fee = 0
        for idx, row in scheme_fraud.iterrows():
            fee = get_fee_for_aci(row, aci, fees_df, mcc)
            if fee is not None:
                total_fee += fee
            else:
                # If no fee structure, assume it's not a valid option or infinite cost?
                # The question implies we can "incentivize users to use a different interaction", implying the ACI is valid.
                # If no fee structure, we can't use it.
                total_fee = float('inf')
                break
        aci_total_fees[aci] = total_fee
    
    # Find the ACI with the minimum total fee
    best_aci = min(aci_total_fees, key=aci_total_fees.get)
    best_fee = aci_total_fees[best_aci]
    
    # Round to 2 decimals
    best_fee_rounded = round(best_fee, 2)
    
    answer = f"{best_scheme}:{best_fee_rounded}"
    print(json.dumps({'answer': answer}))