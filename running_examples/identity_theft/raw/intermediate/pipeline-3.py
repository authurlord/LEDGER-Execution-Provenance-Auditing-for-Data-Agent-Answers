import pandas as pd
import json
import re
import os

# Define base path
base_path = '/mnt/data/yanmy/LakeOrch/KramaBench/data/legal/input/csn-data-book-2024-csv/CSVs'

# 1. Load the main identity theft reports file
# This file contains the list of MSAs and their reports.
# We need to extract MSA names and report counts.
main_file = os.path.join(base_path, '2024_CSN_Metropolitan_Areas_Identity_Theft_Reports.csv')

# Handle encoding error by trying different encodings
try:
    df_main = pd.read_csv(main_file, header=None, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_main = pd.read_csv(main_file, header=None, encoding='latin-1')
    except UnicodeDecodeError:
        df_main = pd.read_csv(main_file, header=None, encoding='cp1252')

# The data starts at row 1 (index 1) with headers "Rank", "Metropolitan Area", "Reports per 100K Population", "# of Reports"
# Let's clean the data
df_main.columns = ['Rank', 'Metropolitan Area', 'Reports per 100K Population', '# of Reports']
df_main = df_main.iloc[1:] # Skip header row if it was included in the snippet logic, but here header is row 1.
# Actually, looking at the snippet:
# Row 0: empty
# Row 1: Rank, Metropolitan Area, ...
# So we should start from row 1.
df_main = df_main.iloc[1:].reset_index(drop=True)

# Clean up columns
df_main['Metropolitan Area'] = df_main['Metropolitan Area'].astype(str).str.strip()
df_main['# of Reports'] = df_main['# of Reports'].astype(str).str.replace(',', '').str.strip()
df_main['# of Reports'] = pd.to_numeric(df_main['# of Reports'], errors='coerce')

# Clean up Reports per 100K Population column
df_main['Reports per 100K Population'] = df_main['Reports per 100K Population'].astype(str).str.replace(',', '').str.strip()
df_main['Reports per 100K Population'] = pd.to_numeric(df_main['Reports per 100K Population'], errors='coerce')

# 2. Load all state-specific identity theft files to ensure we have all MSAs or to cross-reference if needed.
# However, the main file seems to be a comprehensive list. The prompt asks to "Drop entries where there's no match in the html for the areas fraud reports".
# This implies we need to check against "Fraud and Other data" files to see if the MSA exists there.
# Let's collect all MSAs from the fraud files.

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

valid_msa_names = set()

for f in fraud_files:
    fpath = os.path.join(base_path, f)
    if not os.path.exists(fpath):
        continue
    try:
        df_fraud = pd.read_csv(fpath, header=None, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_fraud = pd.read_csv(fpath, header=None, encoding='latin-1')
        except UnicodeDecodeError:
            df_fraud = pd.read_csv(fpath, header=None, encoding='cp1252')
            
    # Structure: Row 0 empty, Row 1 headers, Row 2+ data
    # Headers are "Metropolitan Area", "# of Reports"
    df_fraud = df_fraud.iloc[1:]
    df_fraud.columns = ['Metropolitan Area', '# of Reports']
    
    # Extract MSA names
    msas = df_fraud['Metropolitan Area'].dropna().astype(str).str.strip()
    for msa in msas:
        if msa and msa != 'Metropolitan Area':
            valid_msa_names.add(msa)

# 3. Normalize MSA names for matching
def normalize_msa(name):
    # Ensure name is a string
    if not isinstance(name, str):
        return ""
    # Remove "Metropolitan Statistical Area", "Micropolitan Statistical Area", "MSA"
    name = re.sub(r'\s*(Metropolitan Statistical Area|Micropolitan Statistical Area|MSA)\s*$', '', name, flags=re.IGNORECASE)
    # Remove state suffixes like ", DC-VA-MD-WV" or ", FL" if we want to just use city/state part?
    # The prompt says: "Use only the city and state portion of the name, ignoring suffixes like 'Metropolitan Statistical Area' or 'MSA' and normalizing punctuation."
    # It doesn't explicitly say to remove the state codes (e.g., FL, TX). It says "city and state portion".
    # Usually, MSA names are like "New York-Newark-Jersey City, NY-NJ".
    # Let's keep the core name but strip the suffix.
    name = name.strip()
    return name

# Apply normalization to valid_msa_names
normalized_valid_msas = set()
for msa in valid_msa_names:
    normalized_valid_msas.add(normalize_msa(msa))

# Apply normalization to main df
df_main['Normalized MSA'] = df_main['Metropolitan Area'].apply(normalize_msa)

# Filter main df to only include MSAs that are in the fraud reports
df_main = df_main[df_main['Normalized MSA'].isin(normalized_valid_msas)]

# 4. Get Population Data
# The prompt asks for 2023 population. It says "If you don't have their population size in 2023, use two years where you know the censuses... and linearly interpolate".
# The data files mention "population estimates are based on 2023 U.S. Census figures" in the footer.
# However, the CSVs provided do NOT contain population columns.
# We need to find population data. The prompt implies we might need to look elsewhere or use external knowledge if not in files.
# But wait, the prompt says "Report the average number of reported identity thefts...". It doesn't ask us to calculate the rate, just the average number of reports.
# BUT, the condition is "for all metropolitan areas that are larger than one million in population in 2023".
# So we DO need population data to filter the MSAs.
# Since the provided CSVs don't have population, and the prompt mentions using census data, I must assume I need to use external population data or the prompt implies the population data is available somewhere.
# Looking at the file list, there is no population file.
# However, often in these tasks, if population isn't in the file, we might need to use a standard dataset or the prompt expects us to know that the "Reports per 100K" column can be used to back-calculate population if we have the total reports?
# No, we have "# of Reports" and "Reports per 100K".
# Population = (# of Reports / Reports per 100K) * 100,000.
# This gives us the 2023 population estimate based on the data book's own figures!
# Let's verify this logic.
# The footer says: "population estimates are based on 2023 U.S. Census figures."
# So, using the formula: Pop = (Reports / Rate) * 100,000 should give us the 2023 population used by the FTC.

# Ensure numeric types before calculation
df_main['# of Reports'] = pd.to_numeric(df_main['# of Reports'], errors='coerce')
df_main['Reports per 100K Population'] = pd.to_numeric(df_main['Reports per 100K Population'], errors='coerce')

df_main['Population'] = (df_main['# of Reports'] / df_main['Reports per 100K Population']) * 100000

# 5. Filter for Population > 1,000,000
df_large_msa = df_main[df_main['Population'] > 1000000]

# 6. Calculate Average Identity Theft Reports
if len(df_large_msa) > 0:
    avg_reports = df_large_msa['# of Reports'].mean()
else:
    avg_reports = None

# Round to 4 decimal places
if avg_reports is not None:
    avg_reports = round(avg_reports, 4)

# 7. Prepare result
result = {
    "main-task": avg_reports
}

print(json.dumps(result, indent=4))