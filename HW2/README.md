# NTU ADL 2025 — HW2

## Overview

This repository contains code for a bidirectional translation task between Classical Chinese (文言文) and Modern Chinese (白話文). The project includes scripts for fine-tuning a large language model with QLoRA/PEFT, running inference with a trained adapter, and evaluating model perplexity on test sets.

## Repository structure

```
/
│
├── download.sh        # Download adapter/checkpoint used by run.sh
├── inference.py       # Inference / generation script
├── report.pdf         # Experiment report and configuration details
├── run.sh             # Simple wrapper to run inference
├── test.sh            # Environment & smoke test script
├── utils.py           # Prompt builders and quantization helpers
├── parse_log.py       # Parse training logs to CSV
├── ppl.py             # Perplexity evaluation script (given grading function)
└── tuning.py          # Fine-tuning script (QLoRA + PEFT)
```

## Quick start

1) Download the adapter (if provided):

```bash
./download.sh
```

2) Run inference (example):

```bash
./run.sh <base_model> <adapter_path> <input_json> <output_json>

# or call inference.py directly:
python inference.py --model_path Qwen/Qwen3-4B --adapter_path ./adapter_checkpoint --input_path public_test.json --output_path predictions.json
```

3) Compute perplexity on a test set:

```bash
python ppl.py --base_model_path Qwen/Qwen3-4B --peft_path ./adapter_checkpoint --test_data_path public_test.json
```

4) Fine-tune a model with QLoRA + LoRA (example):

```bash
python tuning.py --base_model Qwen/Qwen3-4B --train_data train.json --output_dir ./adapter_checkpoint --per_device_train_batch_size 1 --num_train_epochs 3
```

Notes:
- `inference.py` uses prompt helpers from `utils.py` to format task instructions and examples.
- `ppl.py` evaluates perplexity by loading the base quantized model and the PEFT adapter.
- `tuning.py` contains the training loop and QLoRA configuration details.

## Scripts

- `download.sh` — download model adapter or other required files.
- `run.sh` — convenience wrapper for `inference.py`.
- `test.sh` — environment check and smoke tests. Useful for quick verification of dependencies and scripts.

## Implementation notes

- Quantization and bits-and-bytes configuration live in `utils.get_bnb_config`.
- Prompt templates and any instruction/prefix formatting are in `utils.get_prompt` (and related functions).
- Training uses PEFT/LoRA combined with k-bit quantization (refer to `tuning.py` for exact hyperparameters used).

## Reported results

| Experiment   | Pre-trained Model              | Epoch | Public test perplexity |
|--------------|--------------------------------|-------|------------------------|
| Experiment 1 | Qwen/Qwen3-4B                  | 5.0   | 6.73                   |
| Experiment 2 | yentinglin/Llama-3.1-Taiwan-8B | 1.0   | 6.12                   |

See `report.pdf` for full configuration and experimental details.

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch (matching your CUDA or CPU setup)
- transformers, datasets, accelerate, peft, bitsandbytes, tqdm, numpy, pandas

Install common dependencies with pip:

```bash
pip install torch transformers datasets accelerate peft bitsandbytes tqdm numpy pandas
```

## Test script
The repository includes a helper script `test.sh` that performs an end-to-end environment and smoke test. It is a Bash script intended for Unix-like environments (Ubuntu, WSL, Git Bash). The script performs the following steps:

- Step 1 — Environment check
- Step 2 — Package check
- Step 3 — Run `download.sh`
- Step 4 — Run `run.sh`
- Step 5 — Compute perplexity with `ppl.py`

Behavior and notes:

- The script prints expected package versions and attempts to generate a `my_requirements.txt` using `pipreqs`.
- It runs `download.sh` and then `run.sh`, capturing logs to `download.log` and `run.log` in the same directory.
- There are commented lines that can simulate no-internet using `iptables` (requires `sudo`) — disabled by default.
- The script assumes `bash`, `python3`, and common Unix utilities (`uname`, `free`, `df`, `awk`) are available. On GPU machines the script will try to use `nvidia-smi` to show GPU info if present.

How to run

```bash
./test.sh
```

Before running

- Edit the top of `test.sh` to set the runtime variables (for example `MODEL_PATH`, `PEFT_PATH`, `PUBLIC_FILE`, and `OUTPUT_FILE`).
- Make sure `download.sh` and `run.sh` are present and executable in the same folder.
- If you run on a GPU machine, ensure `nvidia-smi` is available to show GPU info.

Files produced by the test

- `download.log` — stdout/stderr from `download.sh`
- `run.log` — stdout/stderr from `run.sh`

## References
* Large Language Models
  * Qwen Team. (2025). *Qwen3 Technical Report*. *arXiv preprint* arXiv:2505.09388. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
  * Yen-Ting Lin and Yun-Nung Chen. (2023). *Taiwan LLM: Bridging the Linguistic Divide with a Culturally Aligned Language Model*. *arXiv preprint* arXiv:2311.17487. [https://arxiv.org/abs/2311.17487](https://arxiv.org/abs/2311.17487)

* Finetuning
  * Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. *arXiv preprint* arXiv:2305.14314. [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
  * JianXiao2021. (2024). *Ancient text generation LLM*. GitHub repository. [https://github.com/JianXiao2021/ancient_text_generation_LLM](https://github.com/JianXiao2021/ancient_text_generation_LLM)

