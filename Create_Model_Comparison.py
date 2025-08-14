import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk

OUTPUT_TXT = "Gawikipedia_5_samples.txt"
SOURCES_TXT = "unique_sources.txt"
CACHE_DIR = os.path.join("cache", "gawikipedia")

def is_gawiki(example):
    _id = example.get("id")
    return isinstance(_id, str) and _id.split(":", 1)[0] == "Gawikipedia"

# Load from cache if available
if os.path.isdir(CACHE_DIR):
    ds_gawiki = load_from_disk(CACHE_DIR)
    print(f"Loaded cached dataset from {CACHE_DIR}")
    # Also need ds_all to compute sources if coming from cache; reload minimal metadata
    ds = load_dataset("ReliableAI/Irish-Text-Collection")
    ds_all = concatenate_datasets([ds[split] for split in ds.keys()]) if isinstance(ds, DatasetDict) else ds
else:
    # 1) import dataset
    ds = load_dataset("ReliableAI/Irish-Text-Collection")

    # 2) combine available splits into one dataset
    ds_all = concatenate_datasets([ds[split] for split in ds.keys()]) if isinstance(ds, DatasetDict) else ds

    # 3) filter where id prefix == "Gawikipedia"
    ds_gawiki = ds_all.filter(is_gawiki, num_proc=os.cpu_count())

    # Save to cache for future runs
    os.makedirs(os.path.dirname(CACHE_DIR), exist_ok=True)
    ds_gawiki.save_to_disk(CACHE_DIR)
    print(f"Saved subset to cache: {CACHE_DIR}")

# NEW: collect unique sources (prefixes before ':') across the full dataset
def get_prefix(_id):
    return _id.split(":", 1)[0] if isinstance(_id, str) and ":" in _id else None

# Add a 'source' column, then get uniques (filters out Nones)
ds_with_source = ds_all.map(lambda ex: {"source": get_prefix(ex.get("id"))}, num_proc=os.cpu_count())
unique_sources = sorted([s for s in set(ds_with_source["source"]) if s])

# Save sources to a file
with open(SOURCES_TXT, "w", encoding="utf-8") as f:
    for s in unique_sources:
        f.write(s + "\n")
print(f"Found {len(unique_sources)} unique sources. Wrote to {SOURCES_TXT}")

# 4) get "text" column as list and take 5 samples (first 5)
n = min(5, len(ds_gawiki))
texts = ds_gawiki.select(range(n))["text"]

# 5) save to file separated by \n\n\n
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n\n\n".join(t.strip() for t in texts if isinstance(t, str)))

print(f"Wrote {n} samples to {OUTPUT_TXT}")
