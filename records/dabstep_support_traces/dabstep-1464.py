import pandas as pd
import json

# Load the fees.json file
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# Convert to DataFrame for easier filtering
fees_df = pd.DataFrame(fees_data)

# Filter for account_type = 'R' and aci = 'B'
# Note: account_type is a list in the JSON, so we need to check if 'R' is in the list
# aci is also a list, so we need to check if 'B' is in the list

filtered_fees = fees_df[
    fees_df['account_type'].apply(lambda x: 'R' in x if isinstance(x, list) else False) &
    fees_df['aci'].apply(lambda x: 'B' in x if isinstance(x, list) else False)
]

# Extract the ID values
fee_ids = filtered_fees['ID'].tolist()

# Format the answer as a comma-separated list
if fee_ids:
    answer = ', '.join(map(str, fee_ids))
else:
    answer = ''

print(json.dumps({'answer': answer}))