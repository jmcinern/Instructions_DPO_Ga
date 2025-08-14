# Overview: use both Oireachtas and Wikipedia to seed dataset. 
import pandas as pd
path = "C:/Users/josep/VS-code-projects/Oireachtas_Collect_Analyse/debates_all_2020-01-01_to_2025-01-01.csv"

df = pd.read_csv(path, encoding="utf-8", nrows=100_000)  # Limit to 100k rows for testing

# filter by 'lang' == 'ga'
#df_ga = df[df['lang'] == 'ga']
df_ga_txt = df['text'].tolist()

print(df_ga_txt[:5])  # Print first 5 entries for debugging

