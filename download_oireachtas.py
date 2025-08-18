# URL: https://huggingface.co/datasets/jmcinern/Oireachtas_XML/tree/main
# File:  debates_all_with_lang.csv
# download this file and store in the same directory as this script

import os
import shutil
from pathlib import Path

REPO_ID = "jmcinern/Oireachtas_XML"
FNAME = "debates_all_with_lang.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / FNAME

if OUT_PATH.exists():
    print(f"{OUT_PATH.name} already exists at {OUT_PATH}")
    raise SystemExit(0)

# Try huggingface_hub first (will use HF token if available in env)
try:
    from huggingface_hub import hf_hub_download
    print("Attempting download via huggingface_hub...")
    local = hf_hub_download(repo_id=REPO_ID, filename=FNAME, repo_type="dataset")
    # hf_hub_download returns a path in HF cache — copy to script dir
    shutil.copy(local, OUT_PATH)
    print(f"Downloaded {FNAME} to {OUT_PATH}")
    raise SystemExit(0)
except Exception as e:
    print(f"hf_hub_download failed: {e} — falling back to direct HTTP download")

# Fallback: raw file URL on huggingface (resolve main)
try:
    import requests
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FNAME}"
    print(f"Downloading from {url} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk_size = 1_048_576  # 1MB
        if tqdm and total:
            pbar = tqdm(total=total, unit="B", unit_scale=True)
        else:
            pbar = None
        with open(OUT_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
        if pbar:
            pbar.close()
    print(f"Downloaded {FNAME} to {OUT_PATH}")
except Exception as e:
    print("Download failed:", e)
    raise SystemExit(1)
# filepath: c:\Users\josep\VS-code-projects\Instruction_DPO_Ga\download_oireachtas.py
import os
import shutil
from pathlib import Path

REPO_ID = "jmcinern/Oireachtas_XML"
FNAME = "debates_all_with_lang.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / FNAME

if OUT_PATH.exists():
    print(f"{OUT_PATH.name} already exists at {OUT_PATH}")
    raise SystemExit(0)

# Try huggingface_hub first (will use HF token if available in env)
try:
    from huggingface_hub import hf_hub_download
    print("Attempting download via huggingface_hub...")
    local = hf_hub_download(repo_id=REPO_ID, filename=FNAME, repo_type="dataset")
    # hf_hub_download returns a path in HF cache — copy to script dir
    shutil.copy(local, OUT_PATH)
    print(f"Downloaded {FNAME} to {OUT_PATH}")
    raise SystemExit(0)
except Exception as e:
    print(f"hf_hub_download failed: {e} — falling back to direct HTTP download")

# Fallback: raw file URL on huggingface (resolve main)
try:
    import requests
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FNAME}"
    print(f"Downloading from {url} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk_size = 1_048_576  # 1MB
        if tqdm and total:
            pbar = tqdm(total=total, unit="B", unit_scale=True)
        else:
            pbar = None
        with open(OUT_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
        if pbar:
            pbar.close()
    print(f"Downloaded {FNAME} to {OUT_PATH}")
except Exception as e:
    print("Download failed:", e)
    raise SystemExit(1)