import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk

OUTPUT_TXT = "Gawikipedia_5_samples.txt"
SOURCES_TXT = "unique_sources.txt"
CACHE_DIR = os.path.join("cache", "gawikipedia")

def is_gawiki(example):
    _id = example.get("id")
    return isinstance(_id, str) and _id.split(":", 1)[0] == "Gawikipedia"

if os.path.isdir(CACHE_DIR):
    ds_gawiki = load_from_disk(CACHE_DIR)
    print(f"Loaded cached dataset from {CACHE_DIR}")
    ds = load_dataset("ReliableAI/Irish-Text-Collection")
    ds_all = concatenate_datasets([ds[s] for s in ds.keys()]) if isinstance(ds, DatasetDict) else ds
else:
    ds = load_dataset("ReliableAI/Irish-Text-Collection")
    ds_all = concatenate_datasets([ds[s] for s in ds.keys()]) if isinstance(ds, DatasetDict) else ds
    ds_gawiki = ds_all.filter(is_gawiki, num_proc=os.cpu_count())
    os.makedirs(os.path.dirname(CACHE_DIR), exist_ok=True)
    ds_gawiki.save_to_disk(CACHE_DIR)
    print(f"Saved subset to cache: {CACHE_DIR}")

# UNIQUE SOURCES (no map, no schema issues)
ids = [i for i in ds_all["id"] if isinstance(i, str)]
unique_sources = sorted({i.split(":", 1)[0] for i in ids if ":" in i})
with open(SOURCES_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(unique_sources))
print(f"Found {len(unique_sources)} unique sources. Wrote to {SOURCES_TXT}")

# Write 5 samples
n = min(5, len(ds_gawiki))
texts = ds_gawiki.select(range(n))["text"]
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n\n\n".join(t.strip() for t in texts if isinstance(t, str)))
print(f"Wrote {n} samples to {OUTPUT_TXT}")
