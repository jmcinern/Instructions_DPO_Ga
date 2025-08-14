import pandas as pd
# open the first 10_000 rows of csv
path = "~/code/Oireachtas_Collect_Analyse/debates_all_with_lang.csv"
df = pd.read_csv(path, nrows=10_000)
# get subset where lang == ga
df = df[df["lang"] == "ga"]
# get list of text
text_list = df["text"].tolist()
# filter by >100 chars
text_list = [text for text in text_list if len(text) > 100]
print(f"Number of texts: {len(text_list)}")
# save to txt
with open("ga_texts.txt", "w", encoding="utf-8") as f:
    f.writelines("\n".join(text_list))
# manually choose 10 examples