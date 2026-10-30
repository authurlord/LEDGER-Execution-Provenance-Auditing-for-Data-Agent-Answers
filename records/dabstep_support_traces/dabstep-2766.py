import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
merchant_data = pd.read_json('merchant_data.json')
fees = pd.read_json('fees.json')
acquirer_countries = pd.read_csv('acquirer_countries.csv')

# Filter for Rafa_AI in 2023
rafa_payments = payments[(payments['merchant'] == 'Rafa_AI') & (payments['year'] == 2023)]

# Get Rafa_AI's MCC
rafa_merchant_info = merchant_data[merchant_data['merchant'] == 'Rafa_AI']
if rafa_merchant_info.empty:
    print(json.dumps({'answer': 'Not Applicable'}))
else:
    rafa_mcc = rafa_merchant_info.iloc[0]['merchant_category_code']
    
    # Determine intracountry for each transaction
    # intracountry is True if issuing_country == acquirer_country
    rafa_payments['is_intracountry'] = rafa_payments['issuing_country'] == rafa_payments['acquirer_country']
    
    # Function to check if a rule applies to a transaction
    def get_applicable_rule(row, rafa_mcc):
        applicable_rules = []
        for _, rule in fees.iterrows():
            # Check card scheme
            if rule['card_scheme'] != row['card_scheme']:
                continue
            
            # Check is_credit
            if rule['is_credit'] is not None and rule['is_credit'] != row['is_credit']:
                continue
            
            # Check aci
            if rule['aci'] and row['aci'] not in rule['aci']:
                continue
            
            # Check merchant category code
            if rafa_mcc not in rule['merchant_category_code']:
                continue
            
            # Check intracountry
            rule_intra = rule['intracountry']
            if rule_intra is not None:
                # Convert to bool for comparison
                rule_intra_bool = bool(rule_intra)
                if rule_intra_bool != row['is_intracountry']:
                    continue
            
            applicable_rules.append(rule)
        
        return applicable_rules

    # Calculate fees for each transaction
    rafa_payments['fee'] = 0.0
    rafa_payments['rule_matched'] = None
    
    for idx, row in rafa_payments.iterrows():
        rules = get_applicable_rule(row, rafa_mcc)
        if rules:
            # If multiple rules, pick the most specific one?
            # Specificity can be measured by number of non-null constraints
            best_rule = None
            max_specificity = -1
            
            for rule in rules:
                specificity = 0
                if rule['is_credit'] is not None: specificity += 1
                if rule['aci']: specificity += 1
                if rule['intracountry'] is not None: specificity += 1
                # MCC is always checked, so it's a constraint
                
                if specificity > max_specificity:
                    max_specificity = specificity
                    best_rule = rule
            
            if best_rule is not None:
                fee = best_rule['fixed_amount'] + (row['eur_amount'] * best_rule['rate'] / 100)
                rafa_payments.at[idx, 'fee'] = fee
                rafa_payments.at[idx, 'rule_matched'] = best_rule['card_scheme']
        else:
            # No rule matched? This shouldn't happen if data is consistent.
            # If it does, we might need to handle it. For now, assume 0 fee or error.
            pass

    # Aggregate by card scheme
    if rafa_payments['fee'].sum() == 0:
        print(json.dumps({'answer': 'Not Applicable'}))
    else:
        agg = rafa_payments.groupby('card_scheme').agg(
            total_fee=('fee', 'sum'),
            total_amount=('eur_amount', 'sum')
        )
        agg['effective_rate'] = agg['total_fee'] / agg['total_amount']
        
        min_rate_row = agg['effective_rate'].idxmin()
        min_rate = agg.loc[min_rate_row, 'effective_rate']
        
        answer = f"{min_rate_row}:{min_rate:.2f}"
        print(json.dumps({'answer': answer}))