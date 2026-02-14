import pandas as pd

filepath = "/Users/jonaslee/Desktop/cvrj-bed-forecasting/cvrj_dataset.csv"
df = pd.read_csv(filepath)

df['severity_code'] = df['Offense Code'].str[-2].str.upper()
df['class_code'] = pd.to_numeric(df['Offense Code'].str[-1],errors='coerce').fillna(99)

priority_map = {
    'F':1,
    'M':2,
    'A':3,
    'S':4,
    'I':5
}

df['severity_rank'] = df['severity_code'].map(priority_map).fillna(6)

df_sorted = df.sort_values(
    by=['Booking #', 'severity_rank', 'class_code'], 
    ascending=[True, True, True]
)

df_collapsed = df_sorted.drop_duplicates(subset=['Booking #'], keep='first')

df_final = df_collapsed.drop(columns=['severity_code', 'class_code', 'severity_rank'])

df_final.to_csv('cvrj_dataset_v2.csv', index=False)