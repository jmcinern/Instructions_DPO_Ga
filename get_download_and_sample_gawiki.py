import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk


OUTPUT_TXT = "gawiki_5_samples.txt"
CACHE_DIR = os.path.join("cache", "gawiki")

def is_gawiki(example):
    _id = example.get("id")
    return isinstance(_id, str) and _id.split(":", 1)[0] == "gawiki"

# Load from cache if available
if os.path.isdir(CACHE_DIR):
    ds_gawiki = load_from_disk(CACHE_DIR)
    print(f"Loaded cached dataset from {CACHE_DIR}")
else:
    # 1) import dataset
    ds = load_dataset("ReliableAI/Irish-Text-Collection")

    # 2) combine available splits into one dataset
    ds_all = concatenate_datasets([ds[s] for s in ds.keys()]) if isinstance(ds, DatasetDict) else ds

    # 3) filter where id prefix == "gawiki"
    ds_gawiki = ds_all.filter(is_gawiki, num_proc=os.cpu_count())

    # Save to cache for future runs
    os.makedirs(os.path.dirname(CACHE_DIR), exist_ok=True)
    ds_gawiki.save_to_disk(CACHE_DIR)
    print(f"Saved subset to cache: {CACHE_DIR}")

# 4) get "text" column as list and randomly take 10 samples under 1000 chars
ds_short = ds_gawiki.filter(lambda ex: isinstance(ex.get("text", None), str) and len(ex["text"]) < 1000)
ds_short = ds_short.shuffle(seed=42)
n = min(70, len(ds_short))
texts = ds_short.select(range(n))["text"]

# save two samples of 50 texts for test1 and  20 test fortest2
wiki_1 = texts[:50]
wiki_2 = texts[50:70]

# save to folder seed_data, wiki_test1.txt and wiki_test2.txt
os.makedirs("seed_data", exist_ok=True)
with open("seed_data/wiki_test1.txt", "w", encoding="utf-8") as f:
    f.write("\n\n\n".join(wiki_1)) 

with open("seed_data/wiki_test2.txt", "w", encoding="utf-8") as f:
    f.write("\n\n\n".join(wiki_2))
