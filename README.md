# LoRA: Low-Rank Adaptation — Reproduction & Applied Fine-Tuning

Reproduction of **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al., ICLR 2022), verified using the production-standard [Hugging Face PEFT](https://github.com/huggingface/peft) implementation, and applied to two practical fine-tuning tasks: **Text-to-SQL generation** and **Email Reply Generation**.

Both notebooks fine-tune a small, ungated open-source LLM with LoRA on free Google Colab (T4 GPU), and both ship with a Gradio app that compares the **base model** side-by-side against the **LoRA-adapted model** on identical prompts — the same frozen model, one small adapter, two personalities.

---

## Repository Structure

```
.
├── README.md
├── notebooks/
│   ├── LoRA_Text2SQL_Final.ipynb          # Use Case 1 — Text-to-SQL
│   └── LoRA_Email_gen_DUAL_PATH.ipynb     # Use Case 2 — Email Reply Generation
├── adapters/
│   ├── email_reply_lora_enron_adapter.zip       # Email adapter — OLD (trained incl. short/terse replies)
│   └── email_reply_lora_enron_LONG_adapter.zip  # Email adapter — NEW (trained on longer replies only)
└── report/
    ├── LoRA_Text2SQL_Project_Report.docx
    └── Email_Enron_Only_Report.docx
```

> Rename/organize folders to match your actual repo layout — the structure above reflects what each notebook expects by default (see **How to Run** below).

---

## Use Case 1 — Text-to-SQL Generation

Converts a plain-English question + database schema into an executable SQL query.

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-0.5B-Instruct` (4-bit QLoRA) |
| **Dataset** | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78,577 (question, schema, SQL) triples derived from WikiSQL + Spider |
| **LoRA config** | r=16, α=32, dropout=0.05, targets: `q_proj,k_proj,v_proj,o_proj` |
| **Trainable params** | 2,162,688 / 496,195,456 (**0.4359%**) |
| **Training** | 4,000 samples, 1 epoch, ~13.7 min on a free T4 |
| **Adapter size** | 4.4 MB |

### Results (100 held-out examples)

| Metric | Base model | LoRA fine-tuned |
|---|---|---|
| Exact match (accuracy) | 2.0% | **59.0%** |
| Execution validity (SQLite) | 79.0% | **92.0%** |
| Token-level F1 | 64.3% | **93.9%** |

**Reproduces from the paper:** low-rank bypass injection into attention layers, B=0/A=random initialization (ΔW=0 at start, verified via output parity before training), gradient isolation to the adapter only, and lossless merging (`merge_and_unload()`) back into the base weights for zero added inference latency.

---

## Use Case 2 — Professional Email Reply Generation

Generates a professional email reply in one organization's real writing voice, given an incoming email and (optionally) a stated intent.

| | |
|---|---|
| **Base model** | `unsloth/Qwen2.5-1.5B-Instruct` (4-bit, loaded via [Unsloth](https://github.com/unslothai/unsloth)) |
| **Dataset** | [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) (Kaggle) — 517,401 raw emails, filtered to real (incoming → reply) pairs via the `-----Original Message-----` marker |
| **LoRA config** | r=16, α=32, dropout=0.05, targets: `q_proj,k_proj,v_proj,o_proj` |
| **Trainable params** | 4,358,144 / 1,548,072,448 (**0.2815%**) |
| **Adapter size** | ~29 MB |

Two adapters are included, representing an explicit ablation on **reply-length filtering**:

| Adapter | Reply length filter | Training pairs | Notes |
|---|---|---|---|
| `email_reply_lora_enron_adapter.zip` (**OLD**) | ≥150 characters | 16,238 | Includes short/terse replies |
| `email_reply_lora_enron_LONG_adapter.zip` (**NEW**) | ≥250 characters (auto-relaxed from a 350-char target to stay above a 10,000-pair floor) | 10,478 | Terse replies excluded |

### Results (100 held-out examples, greedy decoding, identical settings across all three)

| Version | ROUGE-1 | ROUGE-L | Avg. reply length |
|---|---|---|---|
| Base (fair/clean prompt) | 0.204 | 0.110 | 97.6 words |
| LoRA — OLD adapter | 0.182 | 0.112 | 41.8 words |
| LoRA — NEW adapter | **0.212** | **0.122** | 62.3 words |

*(Reference/human replies average 75.7–83.4 words depending on the held-out split.)*

**Key finding:** excluding terse replies from training measurably reduced the model's tendency to under-shoot reply length (41.8 → 62.3 words, closer to the ~80-word human average) and improved both ROUGE scores over the base model. The clearest evidence, however, is qualitative — LoRA output authentically mirrors Enron's direct, low-filler internal writing style, while the base model tends to include leftover placeholder text (e.g. `[Your Name]`) unless explicitly instructed not to.

**Known limitation:** LoRA learned *style*, not *knowledge* — it occasionally invents specific facts, names, or dates not present in the incoming email. This is an expected boundary of style-based fine-tuning, not a training defect.

---

## How to Run

Both notebooks are designed for **Google Colab, free tier, T4 GPU** (Runtime → Change runtime type → T4 GPU).

### Text-to-SQL notebook
Run all cells top-to-bottom (Runtime → Run all). Total runtime ≈ 60–90 minutes including training and evaluation.

### Email Reply Generation notebook (dual-path)
This notebook supports **two run modes** — always start with a fresh runtime (Runtime → Restart session) and run **Part 0 (Shared Setup)** first, then choose ONE of:

- **🅰️ Path A — Train from scratch**: downloads Enron, builds the dataset, trains a new adapter (~15–70 min).
- **🅱️ Path B — Load both adapters from zip**: skip training; you'll be prompted to upload the two adapter zips from `adapters/` when each cell runs (~2 minutes total).

Either path converges into the shared sections: **Part 3** (loads the OLD adapter), **Part 4** (shared generation function), **Part 5** (scored evaluation — Path A only, since it needs the held-out set built during training), and **Part 6** (the live Gradio app — works after either path).

---

## Tech Stack

`PyTorch` · `transformers` · `peft` · `trl` (SFTTrainer) · `bitsandbytes` (4-bit QLoRA) · `unsloth` (email notebook) · `datasets` · `evaluate` (ROUGE) · `SQLite3` · `Gradio` · `kagglehub`

---

## Design Notes

- **Why PEFT/Unsloth rather than a from-scratch LoRA implementation:** per project guidance, the production-standard library implementation was used. Reproduction of the paper's mechanism is instead demonstrated empirically — inspecting the injected adapter modules, verifying B=0/A=random initialization, confirming output parity at initialization (untrained adapter = no-op), checking gradient isolation to only the adapter's ~0.3–0.4% of parameters, and confirming lossless merge behavior.
- **Why two email adapters, not one:** to make a specific data-cleaning decision (excluding terse replies) directly measurable, rather than asserted. Both adapters share the same base model, LoRA config, and evaluation protocol — the only variable changed is the training data's minimum reply length.
- **Evaluation methodology:** both use cases use a single shared generation function for *both* scoring and the live demo app, with fixed (greedy) decoding throughout, so reported metrics always reflect exactly what the demo shows.

---

## Limitations & Future Scope

- Single-domain coverage: SQL adapter is schema/query-focused; email adapter covers general corporate correspondence only (no customer support, HR, academic, or PM domains in the current adapters).
- Small held-out evaluation sets (100 examples) — sufficient to observe trends, not statistically exhaustive.
- No rank/target-module ablation performed in this repo (r=16 on attention projections was used throughout, based on the paper's own findings that low rank suffices for attention adaptation).
- Enron data contains real personal names and, in places, real personal information; no PII-scrubbing pass was applied. Any production use of the email adapter should scrub identifying details first.
- Natural next steps: rank ablations (r=4/8/16/32), comparison against LoRA successors (DoRA, AdaLoRA) at equal parameter budget, multi-adapter serving (multiple domain adapters swapped on one base model), and permanent deployment (e.g. Hugging Face Spaces) in place of a live Colab session.

---

## References

- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset card
- [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) (Kaggle)
- [Unsloth](https://github.com/unslothai/unsloth) — efficient LoRA/QLoRA training library
