import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk

OUTPUT_TXT = "Gawikipedia_5_samples.txt"
CACHE_DIR = os.path.join("cache", "gawikipedia")

# Load from cache if available
if os.path.isdir(CACHE_DIR):
    ds_gawiki = load_from_disk(CACHE_DIR)
else:
    # 1) import dataset
    ds = load_dataset("ReliableAI/Irish-Text-Collection")

    # 2) combine available splits into one dataset
    if isinstance(ds, DatasetDict):
        ds_all = concatenate_datasets([ds[split] for split in ds.keys()])
    else:
        ds_all = ds  # already a single Dataset

    # 3) filter to only id == "Gawikipedia"
    ds_gawiki = ds_all.filter(lambda ex: ex["id"] == "Gawikipedia")

    # Save to cache for future runs
    os.makedirs(os.path.dirname(CACHE_DIR), exist_ok=True)
    ds_gawiki.save_to_disk(CACHE_DIR)

# 4) get "text" column as list and take 5 samples (first 5)
n = min(5, len(ds_gawiki))
texts = ds_gawiki.select(range(n))["text"]

# 5) save to file separated by \n\n\n
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n\n\n".join(t.strip() for t in texts if isinstance(t, str)))

print(f"Wrote {n} samples to {OUTPUT_TXT}")