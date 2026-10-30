import json
import pandas as pd
import re
import os

# Define base path
base_path = '/mnt/data/yanmy/LakeOrch/KramaBench/data/legal/input/csn-data-book-2024-csv/CSVs'

# Step 1: Load the main metropolitan areas identity theft reports
main_file = os.path.join(base_path, '2024_CSN_Metropolitan_Areas_Identity_Theft_Reports.csv')

# Handle encoding issues by trying different encodings
try:
    df_main = pd.read_csv(main_file, header=None, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_main = pd.read_csv(main_file, header=None, encoding='latin-1')
    except UnicodeDecodeError:
        df_main = pd.read_csv(main_file, header=None, encoding='cp1252')

# Find the row with 'Rank' to identify the header
# The header row contains 'Rank' in the second column (index 1)
# We need to find the first row where column 1 is 'Rank'
try:
    header_row_idx = df_main[df_main[1] == 'Rank'].index[0]
except IndexError:
    # If 'Rank' is not found, try looking for 'Metropolitan Area' in column 1 or 2
    # Based on the snippet, row 1 has 'Rank' in col 0, 'Metropolitan Area' in col 1
    # Let's check column 0 as well
    header_row_idx = df_main[df_main[0] == 'Rank'].index[0]

df_main = df_main.iloc[header_row_idx+1:]
df_main.columns = ['Rank', 'Metropolitan Area', 'Reports per 100K', 'Num Reports']

# Clean data
df_main['Num Reports'] = df_main['Num Reports'].astype(str).str.replace(',', '', regex=False)
df_main['Num Reports'] = pd.to_numeric(df_main['Num Reports'], errors='coerce')
df_main['Reports per 100K'] = pd.to_numeric(df_main['Reports per 100K'], errors='coerce')

# Step 2: Load state-specific files to ensure we have all MSAs if any are missing from main list
# However, the main list seems comprehensive. The prompt says "Drop entries where there's no match in the html for the areas fraud reports".
# Wait, the prompt says "Drop entries where there's no match in the html for the areas fraud reports".
# This implies we need to cross-reference with Fraud data? Or just that the MSA must exist in the fraud data?
# Let's look at the fraud data files. They contain the same MSA names.
# The instruction "Drop entries where there's no match in the html for the areas fraud reports" is slightly ambiguous.
# It likely means: Only include MSAs that appear in the fraud reports data as well.
# Let's load one fraud file to see the structure and extract all MSAs from fraud data.
# Actually, since the main file is "Metropolitan Areas Identity Theft Reports", and the state files are also "Identity Theft Reports",
# the main file is likely the aggregate. The state files are breakdowns.
# The "fraud reports" part might refer to the "State MSA Fraud and Other data" files.
# Let's assume we need to check if the MSA exists in the Fraud data.
# Let's collect all MSAs from the Fraud data files.

fraud_msa_names = set()
state_files_identity = [
    'State MSA Identity Theft data/DistrictofColumbia.csv',
    'State MSA Identity Theft data/Wyoming.csv',
    'State MSA Identity Theft data/Illinois.csv',
    'State MSA Identity Theft data/Maryland.csv',
    'State MSA Identity Theft data/Indiana.csv',
    'State MSA Identity Theft data/Missouri.csv',
    'State MSA Identity Theft data/SouthDakota.csv',
    'State MSA Identity Theft data/RhodeIsland.csv',
    'State MSA Identity Theft data/Ohio.csv',
    'State MSA Identity Theft data/Alaska.csv',
    'State MSA Identity Theft data/NorthDakota.csv',
    'State MSA Identity Theft data/Idaho.csv',
    'State MSA Identity Theft data/Montana.csv',
    'State MSA Identity Theft data/Minnesota.csv',
    'State MSA Identity Theft data/Mississippi.csv',
    'State MSA Identity Theft data/NewMexico.csv'
]

# We need to check Fraud data files too.
fraud_files = [
    'State MSA Fraud and Other data/DistrictofColumbia.csv',
    'State MSA Fraud and Other data/Wyoming.csv',
    'State MSA Fraud and Other data/Illinois.csv',
    'State MSA Fraud and Other data/Maryland.csv',
    'State MSA Fraud and Other data/Indiana.csv',
    'State MSA Fraud and Other data/Missouri.csv',
    'State MSA Fraud and Other data/SouthDakota.csv',
    'State MSA Fraud and Other data/RhodeIsland.csv',
    'State MSA Fraud and Other data/Ohio.csv',
    'State MSA Fraud and Other data/Alaska.csv',
    'State MSA Fraud and Other data/NorthDakota.csv',
    'State MSA Fraud and Other data/Idaho.csv',
    'State MSA Fraud and Other data/Montana.csv',
    'State MSA Fraud and Other data/Minnesota.csv',
    'State MSA Fraud and Other data/Mississippi.csv',
    'State MSA Fraud and Other data/NewMexico.csv'
]

def extract_msa_from_fraud_file(filepath):
    try:
        df = pd.read_csv(filepath, header=None, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, header=None, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, header=None, encoding='cp1252')
    # Find header row
    try:
        # The header row has 'Metropolitan Area' in the first column
        header_row_idx = df[df[0] == 'Metropolitan Area'].index[0]
        df_data = df.iloc[header_row_idx+1:]
        msas = df_data[0].dropna().unique()
        return msas
    except:
        return []

all_fraud_msas = set()
for f in fraud_files:
    fp = os.path.join(base_path, f)
    if os.path.exists(fp):
        msas = extract_msa_from_fraud_file(fp)
        all_fraud_msas.update(msas)

# Step 3: Normalize names for matching
def normalize_msa_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip()
    # Remove 'Metropolitan Statistical Area' and 'MSA'
    name = re.sub(r'\s*Metropolitan Statistical Area', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*MSA', '', name, flags=re.IGNORECASE)
    # Normalize punctuation: replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Create a normalized mapping for fraud MSAs
normalized_fraud_msas = set()
for msa in all_fraud_msas:
    normalized_fraud_msas.add(normalize_msa_name(msa))

# Filter main dataframe to only include MSAs that match fraud data
df_main['Normalized MSA'] = df_main['Metropolitan Area'].apply(normalize_msa_name)
df_main = df_main[df_main['Normalized MSA'].isin(normalized_fraud_msas)]

# Step 4: Calculate Population from Reports and Reports per 100K
# Formula: Population = (Num Reports / Reports per 100K) * 100,000
df_main['Population'] = (df_main['Num Reports'] / df_main['Reports per 100K']) * 100000

# Step 5: Filter for Population > 1,000,000
df_large_msa = df_main[df_main['Population'] > 1000000]

# Step 6: Calculate Average Identity Theft Reports
if not df_large_msa.empty:
    avg_reports = df_large_msa['Num Reports'].mean()
else:
    avg_reports = None

# Round to 4 decimal places
if avg_reports is not None:
    avg_reports = round(avg_reports, 4)

result = {
    "main-task": avg_reports
}

print(json.dumps(result, indent=4))