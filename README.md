# CPI-Bench

[![Hugging Face Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/TaobaoTmall-AlgorithmProducts/CPI-benchmark)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](#license)

## Introduction

CPI-Bench is a comprehensive suite of benchmarks designed to evaluate whether an image
generation/editing model is truly capable of handling diverse, real-world, and
knowledge-intensive tasks. It consists of four complementary subsets:

| Benchmark | Description | Data Files |
|---|---|---|
| **CPI-General-Benchmark** | General-purpose image editing tasks covering a wide range of task types | `CPI_general_benchmark/CPI_general_benchmark-*.parquet` |
| **CPI-Practical-Benchmark** | Image editing tasks grounded in everyday, real-life scenarios | `CPI_practical_benchmark/CPI_practical_benchmark-*.parquet` |
| **CPI-Intelligent-Benchmark** | Image editing tasks that require domain knowledge and multi-step reasoning, with reference input image(s) | `CPI_intelligent_benchmark-*.parquet` |
parquet` |

Each sample provides an editing/generation instruction (and, for image-editing tasks,
one or more reference images). Models are expected to produce an output image
accordingly, which is then scored by a VLM-as-Judge (e.g., Gemini) across multiple
quality dimensions.

## ✨ Key Features

- **Four Complementary Subsets**: covers general-purpose editing, life-scenario
  editing, and knowledge-intensive reasoning for both image-editing (i2i) and
  text-to-image (t2i) settings.
- **Multi-Image Input Support**: `source` fields may contain one or multiple
  reference images, supporting complex multi-image editing scenarios.
- **Bilingual Instructions**: Chinese and English instructions are provided for
  every subset, enabling cross-lingual evaluation.
- **Reasoning-Aware Annotations**: the reasoning subsets additionally provide a
  `rationale` field — a reference reasoning trace from instruction to expected
  result, used as *guidance material* (not a hard ground truth) during scoring.
- **VLM-Driven Automatic Evaluation**: each subset ships with a ready-to-use,
  multi-dimension VLM-as-Judge evaluation toolkit (see below).

## ✨ Key Attributes

**CPI-General-Benchmark / CPI-Practical-Benchmark fields:**

| Field | Description |
|---|---|
| `id` | Unique sample ID |
| `task` | Task category, used to select the corresponding scoring prompt template |
| `a_to_b_instructions` | Editing instruction in Chinese |
| `a_to_b_instructions_eng` | Editing instruction in English |
| `target_resolution` | Target output resolution |
| `source` | `List[PIL.Image]` — one or more reference input images |

**CPI-Intelligent-Benchmark fields:**

| Field | Description |
|---|---|
| `id` | Unique sample ID |
| `expert_domain` | Domain category, formatted as `"<domain>-<subtask>"` |
| `a_to_b_instructions` | Editing instruction in Chinese |
| `a_to_b_instructions_eng` | Editing instruction in English |
| `rationale` | Reference reasoning trace (guidance material for scoring; may be empty) |
| `target_resolution` | Target output resolution |
| `source` | `List[PIL.Image]` — one or more reference input images |


## Loading

```python
from datasets import load_dataset

# Load a specific subset directly from the Hub (recommended)
dataset = load_dataset("TaobaoTmall-AlgorithmProducts/CPI-benchmark", "general", split="train")
print(dataset)
print(dataset[0])

# Other available configs: "practical", "intelligent"
dataset = load_dataset("TaobaoTmall-AlgorithmProducts/CPI-benchmark", "intelligent", split="train")

# Alternatively, download the repo manually and load from local parquet files
dataset = load_dataset(
    "parquet",
    data_files="/path/to/local/CPI_general_benchmark/CPI_general_benchmark-*.parquet",
    split="train",
)
```

---

# CPI-Bench - Evaluation Toolkit

An automated evaluation toolkit for image generation/editing models, powered by
VLM-as-Judge (e.g., Gemini). Given a set of model outputs, the toolkit scores each
sample across multiple quality dimensions and produces an aggregated report.

The toolkit is located under [`bench_eval_code/`](bench_eval_code) and provides one
evaluation script per subset:

| Subset | Script | Prompt Config |
|---|---|---|
| CPI-General-Benchmark | `bench_eval_code/eval_general_practical.py --benchmark general` | `bench_eval_code/prompts/general_prompts.json` |
| CPI-Practical-Benchmark | `bench_eval_code/eval_general_practical.py --benchmark practical` | `bench_eval_code/prompts/practical_prompts.json` |
| CPI-Intelligent-Benchmark | `bench_eval_code/eval_intelligent.py` | `bench_eval_code/prompts/intelligent_prompts.json` |

## Scoring Dimensions

**CPI-General-Benchmark / CPI-Practical-Benchmark**

Each `task` type is mapped to a task-specific scoring prompt template (defined in
`general_prompts.json` / `practical_prompts.json`). The judge VLM outputs a score for each
dimension in the format `DimensionName: score`, and the sample's final score is the
arithmetic mean across all dimensions returned for that task.

**CPI-Intelligent-Benchmark** — 3 dimensions, each scored 1.0–5.0:

| Dimension | Weight | What it measures |
|---|---|---|
| Knowledge Reasoning | 45% | Factual/domain-knowledge correctness, fused with `rationale` as reference guidance |
| Visual Quality | 30% | Overall visual/aesthetic quality of the generated result |
| Input Consistency | 25% | Consistency between the result and the reference input image(s) |

The final score is a weighted sum of the three dimensions above. If the Knowledge
Reasoning score is ≤ 2, the final score is additionally multiplied by 0.6 as a
penalty for factual/knowledge errors.


## How It Works

- **General / Practical**: a single VLM call per sample — sends
  `[reference image(s)..., result, scoring prompt]` and parses per-dimension scores
  from the response.
- **Intelligent**: a split-call strategy — one VLM call per dimension
  (Knowledge Reasoning, Visual Quality, and for i2i, Input Consistency), so the judge
  can focus on one aspect at a time for more reliable scoring.

## Input Format

First, generate your model's outputs for each sample. If you are not sure which row
corresponds to which image(s)/instruction, use the export helper first — it
auto-detects the dataset schema (image-editing vs. text-to-image) and works for all
four subsets:

```bash
python bench_eval_code/export_samples.py \
    --dataset_path "/path/to/CPI_general_benchmark/CPI_general_benchmark-*.parquet" \
    --output_dir ./exported_general \
    --lang eng \
    --workers 16
```

This produces:
- `source_images/` — reference input images per sample (skipped entirely for the t2i subset, which has no reference images)
- `samples.jsonl` — per-sample metadata: `sample_index`, `id`, `task`, `instruction`, `rationale` (if present)
- `result_template.jsonl` — a template result file; fill in the `result` field with your model's output path after inference

Then prepare a JSONL file mapping each benchmark sample index to your model's
generated result image:

```jsonl
{"sample_index": 0, "result": "/path/to/result_0.png"}
{"sample_index": 1, "result": "/path/to/result_1.png"}
{"sample_index": 2, "result": "/path/to/result_2.png"}
```

- `sample_index`: the 0-based row index into the loaded HF dataset
- `result`: path to your model's generated image for that sample

## Usage

**CPI-General-Benchmark / CPI-Practical-Benchmark:**

```bash
python bench_eval_code/eval_general_practical.py \
    --benchmark general \
    --dataset_path "/path/to/CPI_general_benchmark/CPI_general_benchmark-*.parquet" \
    --result_jsonl "/path/to/my_results.jsonl" \
    --prompts_json bench_eval_code/prompts/general_prompts.json \
    --output_dir eval_output/my_model_general \
    --api_key "YOUR_API_KEY" \
    --lang eng \
    --workers 8
```

Use `--benchmark practical` and `bench_eval_code/prompts/practical_prompts.json` to evaluate
the Practical benchmark instead.

**CPI-Intelligent-Benchmark:**

```bash
python bench_eval_code/eval_intelligent.py \
    --dataset_path "/path/to/CPI_intelligent_benchmark/CPI_intelligent_benchmark-*.parquet" \
    --result_jsonl "/path/to/my_results_i2i.jsonl" \
    --prompts_json bench_eval_code/prompts/intelligent_prompts.json \
    --output_dir eval_output/my_model_intelligent \
    --api_key "YOUR_API_KEY" \
    --lang eng \
    --workers 8
```


## Output

Each script produces two files in `--output_dir`:

- **`cases.jsonl`** — per-sample scoring details (per-dimension scores + raw VLM responses)
- **`summary.json`** — aggregated scores, broken down by task type / domain / dimension

For the t2i subset, `summary.json` additionally includes:
- `overall_avg_score_pct` — the 1–5 score mapped to a 0–100 percentage scale
- `overall_perfect_rate` — the fraction of samples that achieve the maximum score (5) on both dimensions
- `by_domain` — scores aggregated by top-level domain (the part of `expert_domain` before the `-`)

## Features

- **Resume support**: if evaluation is interrupted, re-running the same command
  will skip already-scored samples (found in `cases.jsonl`) and continue from
  where it left off. Use `--no_resume` to force a full re-run.
- **Multi-key rotation**: pass multiple API keys (comma-separated via `--api_key`)
  to distribute requests across keys and avoid rate limits.
- **Concurrent scoring**: use `--workers` to control parallelism for faster evaluation.
- **Custom VLM endpoint**: any OpenAI-compatible API can be used via `--base_url`
  and `--model`.

## File Structure

```
bench_eval_code/
├── bench_utils.py          # Shared utilities: API key pool, image helpers, retry-wrapped VLM caller
├── eval_general_practical.py    # Evaluation script for General / Practical benchmarks
├── eval_intelligent.py   # Evaluation script for Intelligent benchmark
├── export_samples.py       # Dataset export helper (auto-detects schema, multi-threaded)
└── prompts/
    ├── general_prompts.json
    ├── practical_prompts.json
    └── intelligent_prompts.json
```

## License

CPI-Bench is released under the Creative Commons Attribution–NonCommercial–NoDerivatives
(CC BY-NC-ND 4.0) license.

- ✅ Free for academic research purposes only
- ❌ Commercial use is prohibited

By using this dataset, you agree to comply with the applicable license terms.

## 🖊️ Citation

If you find CPI-Bench useful for your research, please consider citing:

```bibtex
```
