# generate_pairs.py
import os
import re
import csv
import uuid
import time
import json
import random
from pathlib import Path

# --- Provider SDKs (install as needed) ---
from openai import OpenAI
import anthropic
import google.generativeai as genai

# ================== CONFIG (edit here) ==================
N_PER_MODEL_PER_SOURCE = 3   # can change to 20 later if desired
SEED_DIR = Path("./seed_data")
OUT_DIR = Path("./outputs")
OUT_CSV = OUT_DIR / "pairs.csv"

# Models to run
OPENAI_MODELS = ["gpt-5", "gpt-5-mini"]
ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]
GOOGLE_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]

# Prompt template (uses {TEXT})
PROMPT_TEMPLATE = """TASK DESCRIPTION
You are given an Irish text source: {TEXT}
YOUR JOB:
Generate an instruction–response pair based on the provided text.
ALLOWED QUESTION TYPES
Is it true that ...
Explain ...
Describe ...
List the steps ...
REQUIREMENTS
The instruction must clearly incorporate the context from the provided text.
The response must be accurate and entirely in Irish.
Output only the instruction–response pair.
OUTPUT FORMAT
Instruction: <instruction in Irish>
Response: <response in Irish>
"""

# Simple retry settings
MAX_RETRIES = 2
RETRY_SLEEP_SEC = 2.0
RANDOM_SEED = 42
# =======================================================

random.seed(RANDOM_SEED)

def load_secrets(path="./secrets.json"):
    with open(path, "r", encoding="utf-8") as f:
        # your file is a list with one dict
        return json.load(f)[0]

def read_seed_files():
    """
    Reads all .txt in ./seed_data and splits chunks on '\n\n\n'.
    Returns dict source_type -> list[str] of chunks.
    source_type ∈ {"Wiki", "Oireachtas"} inferred from filename.
    """
    if not SEED_DIR.exists():
        raise FileNotFoundError(f"Seed directory not found: {SEED_DIR.resolve()}")

    buckets = {"Wiki": [], "Oireachtas": []}
    for p in sorted(SEED_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunks = [c.strip() for c in text.split("\n\n\n") if c.strip()]
        # Infer source
        fname = p.name.lower()
        if "wiki" in fname:
            buckets["Wiki"].extend(chunks)
        elif "oireachtas" in fname:
            buckets["Oireachtas"].extend(chunks)
        else:
            # default to Wiki if ambiguous (can adjust)
            buckets["Wiki"].extend(chunks)
    return buckets

def build_prompt(chunk_text: str) -> str:
    return PROMPT_TEMPLATE.format(TEXT=chunk_text)

def parse_instruction_response(raw: str):
    """
    Extracts the two fields. Returns (instruction, response) or (None, None).
    Accepts minor formatting noise.
    """
    if not raw:
        return None, None

    # Normalize line endings
    s = raw.strip()

    # Primary regex: Instruction: ... Response: ...
    m = re.search(r'Instruction\s*:\s*(.*?)\n\s*Response\s*:\s*(.*)\Z', s, re.DOTALL | re.IGNORECASE)
    if m:
        instr = m.group(1).strip()
        resp = m.group(2).strip()
        if instr and resp:
            return instr, resp

    # Fallback: try to find lines starting with the tags anywhere
    instr_match = re.search(r'^\s*Instruction\s*:\s*(.+)$', s, re.IGNORECASE | re.MULTILINE)
    resp_match  = re.search(r'^\s*Response\s*:\s*(.+)$', s, re.IGNORECASE | re.MULTILINE)
    if instr_match and resp_match:
        return instr_match.group(1).strip(), resp_match.group(1).strip()

    return None, None

def ensure_outfile():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_CSV.exists():
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "model", "source_type", "instruction", "response"])

def append_row(run_id, model, source_type, instruction, response):
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([run_id, model, source_type, instruction, response])

# ----------------------- API CALLS -----------------------

def call_openai(client: OpenAI, model: str, prompt: str) -> str:
    for attempt in range(1 + MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return r.choices[0].message.content
        except Exception as e:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)

def call_anthropic(anthro_client: anthropic.Anthropic, model: str, prompt: str) -> str:
    for attempt in range(1 + MAX_RETRIES):
        try:
            r = anthro_client.messages.create(
                model=model,
                max_tokens=800,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            # content is a list of blocks; take first text block
            for block in r.content:
                if getattr(block, "type", "") == "text" or hasattr(block, "text"):
                    return getattr(block, "text", "")
            return ""
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)

def call_google(model_client, prompt: str) -> str:
    for attempt in range(1 + MAX_RETRIES):
        try:
            r = model_client.generate_content(prompt)
            return getattr(r, "text", "") or ""
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)

# ----------------------- MAIN LOGIC ----------------------

def main():
    # Load keys
    secrets = load_secrets()
    open_ai_key = secrets.get("open_ai")
    anthropic_key = secrets.get("anthropic")
    google_key = secrets.get("google")

    # Init clients
    openai_client = OpenAI(api_key=open_ai_key)
    anthro_client = anthropic.Anthropic(api_key=anthropic_key)
    genai.configure(api_key=google_key)

    ensure_outfile()
    buckets = read_seed_files()

    # For determinism: pick the first N chunks per source (or random.sample if you prefer)
    selected = {
        "Wiki": buckets["Wiki"][:N_PER_MODEL_PER_SOURCE],
        "Oireachtas": buckets["Oireachtas"][:N_PER_MODEL_PER_SOURCE],
    }

    # ---------- Google ----------
    for model in GOOGLE_MODELS:
        gmodel = genai.GenerativeModel(model)
        for source_type, chunk_list in selected.items():
            for i, chunk in enumerate(chunk_list, 1):
                prompt = build_prompt(chunk)
                raw = call_google(gmodel, prompt)
                instr, resp = parse_instruction_response(raw)
                # If malformed once, try one re-ask
                if not instr or not resp:
                    raw = call_google(gmodel, prompt)
                    instr, resp = parse_instruction_response(raw)
                if instr and resp:
                    rid = f"{uuid.uuid4().hex[:8]}-{model}-{source_type}-{i}"
                    append_row(rid, model, source_type, instr, resp)

    # ---------- OpenAI ----------
    for model in OPENAI_MODELS:
        for source_type, chunk_list in selected.items():
            for i, chunk in enumerate(chunk_list, 1):
                prompt = build_prompt(chunk)
                raw = call_openai(openai_client, model, prompt)
                instr, resp = parse_instruction_response(raw)
                if not instr or not resp:
                    raw = call_openai(openai_client, model, prompt)
                    instr, resp = parse_instruction_response(raw)
                if instr and resp:
                    rid = f"{uuid.uuid4().hex[:8]}-{model}-{source_type}-{i}"
                    append_row(rid, model, source_type, instr, resp)

    # ---------- Anthropic ----------
    for model in ANTHROPIC_MODELS:
        for source_type, chunk_list in selected.items():
            for i, chunk in enumerate(chunk_list, 1):
                prompt = build_prompt(chunk)
                raw = call_anthropic(anthro_client, model, prompt)
                instr, resp = parse_instruction_response(raw)
                if not instr or not resp:
                    raw = call_anthropic(anthro_client, model, prompt)
                    instr, resp = parse_instruction_response(raw)
                if instr and resp:
                    rid = f"{uuid.uuid4().hex[:8]}-{model}-{source_type}-{i}"
                    append_row(rid, model, source_type, instr, resp)

    print(f"Done. Wrote to {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
