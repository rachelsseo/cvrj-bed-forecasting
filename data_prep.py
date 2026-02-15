import pandas as pd

# Update this path to the folder where your CSVs are stored
folder_path = "/Users/jonaslee/Desktop/cvrj-bed-forecasting/cvrj_dataset_2012-25.csv"
output_file = "cvrj_dataset_v2.csv"


df_list = []    
df = pd.read_csv(folder_path)

# Some files have offense_code vs offense code
df.columns = [c.strip().replace('_', ' ').replace('\n', ' ') for c in df.columns]
    
if 'Offense Code' not in df.columns:
    for col in df.columns:
        if 'Offense' in col and 'Code' in col:
            df.rename(columns={col: 'Offense Code'}, inplace=True)
            break
    
df_list.append(df)
# extracts severity char
df['severity_char'] = df['Offense Code'].astype(str).str[-2].str.upper()

# extracts class digit
df['class_digit'] = pd.to_numeric(df['Offense Code'].astype(str).str[-1], errors='coerce').fillna(99)

# defines priority
priority_map = {
    'F': 1, # Felony
    'M': 2, # Misdemeanor
    'A': 3, # Admin
    'S': 4, # Status
    'I': 5  # Infraction
}
df['severity_rank'] = df['severity_char'].map(priority_map).fillna(6)

# sorts by Booking # first to group duplicates.
df_sorted = df.sort_values(
    by=['Booking #', 'severity_rank', 'class_digit'],
    ascending=[True, True, True]
)

# collapses duplicates
df_final = df_sorted.drop_duplicates(subset=['Booking #'], keep='first')

# cleanup
df_final = df_final.drop(columns=['severity_char', 'class_digit', 'severity_rank'])

# drops columns that are completely empty
df_final = df_final.dropna(axis=1, how='all')

# saves to csv
df_final.to_csv(output_file, index=False)