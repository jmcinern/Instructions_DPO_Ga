# path = ~/code/Oireachtas_Collect_Analyse/debates_all_with_lang.csv
# this file is huge so chunk it, also use multiple CPUs
# filter df by lang=="ga"
# get "text" column
# randomly sample 70 texts
# split into two files, 50 and 20 texts
# save to seed_data folder

import os
import random
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Config
# Prefer absolute path; fallback to local file if not found
INPUT_CSV = "./debates_all_with_lang.csv"
SEED = 42
CHUNKSIZE = 100_000
SAMPLE_SIZE = 160
MAX_CHARS = 1000  # keep texts under 1000 chars (like gawiki_sample.py)
SEED_DIR = "seed_data"
OUT1 = os.path.join(SEED_DIR, "oireachtas_test1.txt")  # 120 texts
OUT2 = os.path.join(SEED_DIR, "oireachtas_test2.txt")  # 40 texts

random.seed(SEED)
os.makedirs(SEED_DIR, exist_ok=True)

def filter_chunk(df: pd.DataFrame) -> list[str]:
    # Keep only Irish (ga), non-empty strings, and length constraint
    df = df[df["lang"] == "ga"]
    df = df["text"].dropna()
    out = []
    for t in df:
        if isinstance(t, str):
            t = t.strip()
            if t and len(t) <= MAX_CHARS:
                out.append(t)
    return out

def reservoir_update(reservoir: list[str], candidates: list[str], k: int, seen: int) -> int:
    # Standard reservoir sampling across a stream
    for t in candidates:
        seen += 1
        if len(reservoir) < k:
            reservoir.append(t)
        else:
            j = random.randint(1, seen)
            if j <= k:
                reservoir[j - 1] = t
    return seen

def main():
    reservoir: list[str] = []
    seen = 0

    # Stream-read in chunks and process in parallel
    chunk_iter = pd.read_csv(
        INPUT_CSV,
        usecols=["lang", "text"],
        chunksize=CHUNKSIZE,
        encoding="utf-8",
        low_memory=True,
    )

    with ProcessPoolExecutor() as ex:
        futures = []
        for chunk in chunk_iter:
            futures.append(ex.submit(filter_chunk, chunk))

        for fut in as_completed(futures):
            texts = fut.result()
            if texts:
                seen = reservoir_update(reservoir, texts, SAMPLE_SIZE, seen)

    # Final sampled texts
    sampled = reservoir[:SAMPLE_SIZE]
    # Split into 50 and 20
    part1 = sampled[:120]
    part2 = sampled[120:160]

    with open(OUT1, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(part1))
    with open(OUT2, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(part2))

    print(f"Input: {INPUT_CSV}")
    print(f"Seen GA texts: {seen}, sampled: {len(sampled)}")
    print(f"Wrote {len(part1)} to {OUT1}")
    print(f"Wrote {len(part2)} to {OUT2}")

if __name__ == "__main__":
    main()