# generate_pairs.py
# JSON-native, using OpenAI structured outputs, Anthropic tools (input_schema),
# and the new Google GenAI SDK (google-genai) with GenerateContentConfig.

import csv
import json
import random
import time
import uuid
from pathlib import Path
from typing import Optional, Dict

# --- Provider SDKs (install as needed) ---
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types

# ================== CONFIG (edit here) ==================
N_PER_MODEL_PER_SOURCE = 2  
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
QUESTION TYPES:
Is it true that ...
Explain ...
Describe ...
List the steps ...
Translate from Irish to English ...
Translate from English to Irish ...
REQUIREMENTS
Extract facts that are stated/directly implied from the given text.
The response must be accurate and entirely in Irish.
Based on the extracted fact(s), create instuction fine-tuning pair in Irish.
The instructions MUST incorporate the provided
 context where relevant to make the questions
 more specific and meaningful.
OUTPUT FORMAT (STRICT):
Return strict JSON with exactly:
{{
  "instruction": "<instruction in Irish>",
  "response": "<response in Irish>"
}}
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
        fname = p.name.lower()
        if "wiki" in fname:
            buckets["Wiki"].extend(chunks)
        elif "oireachtas" in fname:
            buckets["Oireachtas"].extend(chunks)
        else:
            buckets["Wiki"].extend(chunks)
    return buckets


def build_prompt(chunk_text: str) -> str:
    return PROMPT_TEMPLATE.format(TEXT=chunk_text)


def ensure_outfile():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_CSV.exists():
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "model", "source_type", "instruction", "response", "text"])


def append_row(run_id, model, source_type, instruction, response, chunk):
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([run_id, model, source_type, instruction, response, chunk])


# ----------------------- Structured Output Schemas -----------------------
# OpenAI structured outputs (JSON schema)
INSTRUCTION_PAIR_SCHEMA_DICT = {
    "name": "InstructionPair",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "instruction": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["instruction", "response"]
    },
    "strict": True
}

# Anthropic tool schema
ANTHROPIC_TOOL = {
    "name": "record_instruction_pair",
    "description": "Return an instruction and an Irish response as JSON (InstructionPair).",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "instruction": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["instruction", "response"]
    }
}


# ----------------------- Provider Calls (return dict) -----------------------
def call_openai(client: OpenAI, model: str, prompt: str) -> Optional[Dict[str, str]]:
    for attempt in range(1 + MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                response_format={"type": "json_schema", "json_schema": INSTRUCTION_PAIR_SCHEMA_DICT},
            )
            msg = r.choices[0].message
            data = getattr(msg, "parsed", None)
            if data is None:
                data = json.loads(msg.content)
            if isinstance(data, dict) and data.get("instruction") and data.get("response"):
                return data
            return None
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)


def call_anthropic(anthro_client: anthropic.Anthropic, model: str, prompt: str) -> Optional[Dict[str, str]]:
    for attempt in range(1 + MAX_RETRIES):
        try:
            r = anthro_client.messages.create(
                model=model,
                temperature=1,
                tools=[ANTHROPIC_TOOL],
                tool_choice={"type": "tool", "name": "record_instruction_pair"},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            # Find the tool_use block and return its input as dict
            for block in r.content:
                if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == "record_instruction_pair":
                    data = getattr(block, "input", {}) or {}
                    if isinstance(data, dict) and data.get("instruction") and data.get("response"):
                        return data
            return None
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)


def call_google(client: genai.Client, model: str, prompt: str):
    obj_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "instruction": types.Schema(type=types.Type.STRING),
            "response":   types.Schema(type=types.Type.STRING),
        }
    )
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=obj_schema,
    )

    for attempt in range(1 + MAX_RETRIES):
        try:
            r = client.models.generate_content(model=model, contents=prompt, config=cfg)
            data = json.loads(r.text) if getattr(r, "text", None) else {}
            if isinstance(data, dict) and data.get("instruction") and data.get("response"):
                return data
            return None
        except Exception:
            if attempt >= MAX_RETRIES: raise
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
    google_client = genai.Client(api_key=google_key)

    ensure_outfile()
    buckets = read_seed_files()

    # Pick the first N chunks per source
    selected = {
        "Wiki": buckets["Wiki"][:N_PER_MODEL_PER_SOURCE],
        "Oireachtas": buckets["Oireachtas"][:N_PER_MODEL_PER_SOURCE],
    }

    # ---------- Google ----------
    for model in GOOGLE_MODELS:
        for source_type, chunk_list in selected.items():
            for i, chunk in enumerate(chunk_list, 1):
                prompt = build_prompt(chunk)
                data = call_google(google_client, model, prompt) or call_google(google_client, model, prompt)
                if data:
                    rid = f"{uuid.uuid4().hex[:8]}-{model}-{source_type}-{i}"
                    append_row(rid, model, source_type, data["instruction"], data["response"], chunk)

    # ---------- OpenAI ----------
    for model in OPENAI_MODELS:
        for source_type, chunk_list in selected.items():
            for i, chunk in enumerate(chunk_list, 1):
                prompt = build_prompt(chunk)
                data = call_openai(openai_client, model, prompt) or call_openai(openai_client, model, prompt)
                if data:
                    rid = f"{uuid.uuid4().hex[:8]}-{model}-{source_type}-{i}"
                    append_row(rid, model, source_type, data["instruction"], data["response"], chunk)

    # ---------- Anthropic ----------
    for model in ANTHROPIC_MODELS:
        for source_type, chunk_list in selected.items():
            for i, chunk in enumerate(chunk_list, 1):
                prompt = build_prompt(chunk)
                data = call_anthropic(anthro_client, model, prompt) or call_anthropic(anthro_client, model, prompt)
                if data:
                    rid = f"{uuid.uuid4().hex[:8]}-{model}-{source_type}-{i}"
                    append_row(rid, model, source_type, data["instruction"], data["response"], chunk)

    print(f"Done. Wrote to {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
