## Instruction_DPO_Ga (Model Ranking & Experiment Submodule)

Pipeline for creating Irish (Gaeilge) instruction & preference data via:
- Sampling Oireachtas debates + GaWiki text
- Multi‑model instruction/response synthesis (OpenAI / Anthropic / Google etc.)
- Human + LLM pairwise annotation (A/B)
- Bradley–Terry ranking to determine best model for Irish text generation.
- Analyse interannotator agreement statistics, to evaluate LLM/Learner capabilities in comparison to native.
- (Planned) Use the winning model to generate Irish instruction tuning data.

### Goals
1. Select best base generator (model + price tier) for large scale Irish data synthesis.
2. Compare reference sources (Oireachtas vs Wikipedia) for instruction grounding quality.
3. Validate substitutability of Learner / LLM annotators vs Native speakers (agreement metrics).

### Core Scripts
| Script | Purpose |
|--------|---------|
| `download_oireachtas.py` | Fetch debate CSV (with language column) from Hugging Face. |
| `gawiki_sample.py` | Cache + sample GaWiki subset (id prefix filter) into seed text files. |
| `oireachtas_sample.py` | Reservoir sample Irish debate lines (len ≤1000) into test splits. |
| `Create_Model_Comparison.py` | Generate instruction–response rows across models; logs CSV (now with `source_text`). |
| `gpt4o_annotation.py` | Automated LLM pair annotation (A/B). |
| `human_feedback.py` | Gradio UI for human pairwise annotation (remove deprecated `sharing=` param). |
| `Bradley_Terry.py` | Bradley–Terry ranking + win probability matrices + (optional) kappa. |
| `DPO.py` | Placeholder for Direct Preference Optimization training stage. |

### Data Flow Overview
1. Acquire debate data (`download_oireachtas.py`).
2. Sample GaWiki + Oireachtas seed corpora (`gawiki_sample.py`, `oireachtas_sample.py`).
3. Generate model outputs (`Create_Model_Comparison.py`) → CSV with per‑row `instruction`, `response`, `source_text`.
4. Construct comparison pairs + annotate (`gpt4o_annotation.py`, `human_feedback.py`).
5. Aggregate & rank (`Bradley_Terry.py`).
6. (Future) Train DPO model using ranked preferences (`DPO.py`).

### Annotation Strategy
- Annotator types: Native, Learner, GPT‑4o (LLM), Tester (internal/debug).
- A/B comparisons recorded with choice (A/B) plus metadata.
- Bradley–Terry scores -> win probabilities -> ordering.
- Agreement: Cohen’s kappa / pairwise alignment to justify scalable annotators.

### Prompt Template (Synthesis)
```
You are given an Irish text source: {TEXT}
Generate an instruction–response pair grounded ONLY in that text.
Allowed question styles: “Is it true that…”, “Explain…”, “Describe…”, “List the steps…”.
Both instruction and response must be fully in Irish and factually consistent.
Format exactly:
Instruction: <instruction in Irish>
Response: <response in Irish>
```

### Sampling Parameters
- GaWiki: Filter id prefix `Gawiki`, length <1000 chars, shuffled, two splits (test1/test2).
- Oireachtas: Reservoir sample 70 Irish (`lang == ga`) lines (len ≤1000) into 50/20 splits.

### Generated CSV Columns
`run_id, model, source_type, source_text, instruction, response`

### Bradley–Terry Notes
- Convert A/B pairs to win–loss list.
- Fit via `choix.opt_pairwise` (logistic BT).
- Produce probability matrix P(i beats j).
- Skip sparse annotators (< threshold) if desired.

### Gradio App Note
If encountering: `TypeError: Blocks.launch() got an unexpected keyword argument 'sharing'` → remove `sharing=` and use `share=True` only when needed.

### SLURM (`slurm.sh`)
Template for cluster execution; edit partition, time, memory, env activation. Typical usage:
```bash
sbatch slurm.sh
```

