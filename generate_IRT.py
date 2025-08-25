'''
Overview: create first of its kind instruction-tuning dataset for Irish using Gemini-2.5
1) 800 LIMA/100 Oireachtas/100 Wiki

'''
# Use LIMA for seeding the Oireachtas and Wiki Questions ./LIMA.jsonl
import json
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import time
from typing import Dict, List, Optional, Tuple
import random
import os, json, time, random, hashlib, unicodedata, re
from tqdm import tqdm
import argparse
import asyncio

# limit for parsing for testing, then DPO subset, before full trans.
p = argparse.ArgumentParser()
p.add_argument("-n","--num", type=int, help="Max pairs to translate")
args = p.parse_args()


MAX_RETRIES = 2
RETRY_SLEEP_SEC = 2.0
RANDOM_SEED = 42

# limit number of thinking tokens for each request, default: 8192, max: 32768
THINK_BUDGET = 8192 # keep default for now.

random.seed(RANDOM_SEED)

# helpers to allow for deterministic hashing
def normalize_text(s):
    # Trim, collapse newlines a bit, NFC normalize
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    return s

def stable_hash(instruction, response):
    # Deterministic digest of normalized pair
    instr = normalize_text(instruction)
    resp = normalize_text(response)
    payload = "\x1e".join([instr, resp]).encode("utf-8")  # record-separator join
    return hashlib.sha256(payload).hexdigest()


# translate EN=>GA prompt
translation_prompt =    '''
Translate the following English Instruction and prompt into Irish. response1 should be a direct translation,
whereas response2 should be equally valid but differ in tone, meaning & syntax. It should be hard for a native Irish speaker to 
pick the better answer.
OUTPUT FORMAT (STRICT):
Return strict JSON with exactly:
{{
"instruction": "<instruction in Irish>",
"response1": "<response in Irish>",
"response2": "<response in Irish>"
}}
The following is the English prompt-response pair: 
'''

output_format_reminder ='''OUTPUT FORMAT (STRICT):
Return strict JSON with exactly:
{{
"instruction": "<instruction in Irish>",
"response1": "<response in Irish>",
"response2": "<response in Irish>"
}}'''

# Use LIMA for seeding the Oireachtas and Wiki Questions ./LIMA.jsonl
IRT_ga = []
with open("LIMA.jsonl", "r", encoding="utf-8") as f:
    for ln, raw in enumerate(f, 1):
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)                       # parse the JSON line
        conv = obj.get("conversations", [])
        if len(conv) >= 2 and isinstance(conv[0], str):
            prompt, response = conv[0], conv[1]     # two-string format
        elif len(conv) >= 2 and isinstance(conv[0], dict):
            # role-based mirrors (rare): pick text/content/value
            get = lambda d: d.get("value") or d.get("content") or d.get("text") or ""
            prompt, response = get(conv[0]), get(conv[1])
        else:
            raise ValueError(f"Line {ln}: unexpected conversations format")

        IRT_ga.append({"instruction": prompt, "response": response, "hash": stable_hash(prompt, response)})

# to call Google API
def gemini_trans(model: GenerativeModel, pair_en: dict, prompt: str) -> Optional[str]:
    instruction_en = pair_en.get("instruction", "")
    response_en = pair_en.get("response", "")
    prompt = prompt + "\n\n" + "\n instruction_en: \n" + instruction_en + "\n response_en: \n" + response_en + "\n Reminder of output format: \n" + output_format_reminder
    try:
        response = model.aio.models.generate_content(prompt)
        print(f"Gemini translation response: {response}")
        return response.text or None

    except:
        print(f"Gemini translation failed")
        return None
    

model = GenerativeModel('gemini-2.5-pro')
file_name = "translated_IRT_ga.jsonl"

# allow rerunning of pipeline buy hasing, read with append mode 
already_translated_hashes = []
with open(file_name, "r", encoding="utf-8") as f:
    already_translated_hashes = set()
    for line in f:
        obj = json.loads(line)
        already_translated_hashes.add(obj.get("hash"))

# IRT = IRT - already_translated_hashes
IRT_ga = [irt for irt in IRT_ga if irt.get("hash") not in already_translated_hashes]

gemini_project_id = "gen-lang-client-0817118952" 
gcloud_location = "us-central1"
vertexai.init(project=gemini_project_id, location=gcloud_location)


with open (file_name, "a", encoding="utf-8") as f:  
    to_process = IRT_ga[:args.num] if args.num else IRT_ga
    for IRT in tqdm(to_process):
        translated_IRT_ga = gemini_trans(model, IRT, translation_prompt)
        if translated_IRT_ga is None:
            print("empty response")
        else:
            try:
                # gemini loves wrapping in markdown but can't parse that json directly.
                translated_IRT_ga_clean = translated_IRT_ga.strip().removeprefix('```json\n').removesuffix('```')
                translated_IRT_ga_json = json.loads(translated_IRT_ga_clean)
                # add original en prompt, response to line
                translated_IRT_ga_json["instruction_en"] = IRT["instruction"]
                translated_IRT_ga_json["response_en"] = IRT["response"]
                # add hash for en pair to save in ouput file for cross-referencing.
                hash_en = stable_hash(IRT["instruction"], IRT["response"])
                translated_IRT_ga_json["hash"] = hash_en

                f.write(json.dumps(translated_IRT_ga_json, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"JSON parse error: {e}")
                print(f"Original response: {translated_IRT_ga}")



