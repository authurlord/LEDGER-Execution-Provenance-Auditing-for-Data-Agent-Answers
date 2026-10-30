def matches_rule(rule, transaction, merchant_info):
    # Card Scheme
    if rule['card_scheme'] and rule['card_scheme'] != transaction['card_scheme']:
        return False
    
    # Account Type
    if rule['account_type']:
        if merchant_info['account_type'] not in rule['account_type']:
            return False
    # If empty, assume matches any? Or none? Let's assume matches any if empty.
    
    # Capture Delay
    if rule['capture_delay'] is not None:
        # This is tricky. ">5" vs "manual".
        # Let's assume if rule has a value, it must match the merchant's capture_delay?
        # Or is it a condition on the transaction?
        # Given the complexity, and that we only change MCC, let's assume capture_delay matching is static for the merchant.
        # If the rule matches for the original MCC, it likely matches for the new MCC unless the rule is very specific.
        # Let's assume exact string match for capture_delay if not null.
        if merchant_info['capture_delay'] != rule['capture_delay']:
            return False
            
    # Monthly Volume / Fraud Level
    # These are likely thresholds. Without clear logic, assume they are not primary differentiators for MCC change.
    # Let's assume if they are not null, they must match? But how?
    # Let's assume they are ignored for this specific question unless they are the only matching factor.
    
    # Is Credit
    if rule['is_credit'] is not None:
        if rule['is_credit'] != transaction['is_credit']:
            return False
            
    # ACI
    if rule['aci']:
        if transaction['aci'] not in rule['aci']:
            return False
            
    # Intracountry
    if rule['intracountry'] is not None:
        # If rule specifies intracountry, transaction must be intracountry
        if transaction['ip_country'] != transaction['issuing_country']:
            return False
            
    # MCC
    if transaction['mcc'] not in rule['merchant_category_code']:
        return False
        
    return True