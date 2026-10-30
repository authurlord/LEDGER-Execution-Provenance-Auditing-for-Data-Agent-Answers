import json
    import pandas as pd

    with open('fees.json', 'r') as f:
        fees = json.load(f)

    df = pd.DataFrame(fees)

    # Filter out rows where is_credit is null
    df_valid = df.dropna(subset=['is_credit'])

    # Group by is_credit and calculate mean rate
    grouped = df_valid.groupby('is_credit')['rate'].mean()
    print(grouped)