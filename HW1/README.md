# NTU ADL 2025 — HW1

## Overview

This repository contains code and notebooks for a two-stage Chinese extractive question-answering (QA) system. The pipeline is split into:

- Paragraph Selection (PS): choose the most relevant paragraph (4 candidates) for a given question.
- Span Selection (SS): extract the exact answer span from the selected paragraph.

The code implements inference for Experiment 2 (using a macBERT base model) and includes notebooks used for training and analysis.

## Repository Layout

```
/
│
├── download.sh           # Script to download models and datasets for run.sh
├── ps_inference.py       # Inference script for the 'Paragraph Selection' model
├── report.pdf            # Experiment report
├── run.sh                # Script to run inference
├── ss_inference.py       # Inference script for the 'Span Selection' model
├── test.sh               # Script to check environment and run test evaluation
└── notebooks/            # All of the Jupyter notebook
    ├── all_inference.ipynb        # Experiment 4 inference
    ├── all_train.ipynb            # Experiment 4 train
    ├── no_pretrain_ps_train.ipynb # Experiment 3 train
    ├── ps_inference.ipynb         # Experiment 1, 2, 3 inference
    ├── ps_train.ipynb             # Experiment 1, 2 train
    ├── ss_inference.ipynb         # Experiment 1, 2, 3 inference
    └── ss_train.ipynb             # Experiment 1, 2 train
```

## Quick Start — Inference (Experiment 2)

1. Download required models and datasets:

```
./download.sh
```

2. Run the full inference pipeline (PS -> SS):

```
./run.sh "path/to/context.json" "path/to/test.json" "path/to/prediction.csv"
```

Notes:

- `run.sh` runs paragraph selection first and writes intermediate outputs (e.g., `ps_result.csv`), then runs span selection to produce the final `prediction.csv`.
- You can run each step separately (examples below).

Paragraph selection only:

```powershell
python ps_inference.py --context_file path/to/context.json --test_file path/to/test.json --output_file ps_result.csv
```

Span selection only (requires context or `ps_result.csv` depending on your workflow):

```powershell
python ss_inference.py --test_file path/to/test.json --output_file prediction.csv
```

## Output Files

- `prediction.csv` — final predictions: columns typically `id, answer`.
- `ps_result.csv` — intermediate paragraph selection results (question -> chosen paragraph id).

## Environment & Requirements

The original experiments were run on Kaggle.

Recommended runtime:

- Python 3.10+
- PyTorch (compatible with your CUDA or CPU setup)
- transformers, datasets, accelerate, pandas, tqdm, numpy

Install dependencies via pip:
```
pip install torch transformers datasets accelerate pandas tqdm numpy
```

## Results Summary

| Experiment   | Task                | Pre-trained Model          | Training Time | Validation Accuracy | Testing Accuracy |
|--------------|---------------------|----------------------------|---------------|---------------------|------------------|
| Experiment 1 | paragraph selection | bert-base-chinese          | 01:59:28      | 95.21%              | 73.80%           |
|              | span selection      | bert-base-chinese          | 02:55:02      | 79.23%              |                  |
| Experiment 2 | paragraph selection | chinese-macbert-base       | 02:03:55      | 95.71%              | 76.84%           |
|              | span selection      | chinese-macbert-base       | 03:08:26      | 81.99%              |                  |
| Experiment 3 | paragraph selection | non pre-trained transformer| 00:02:17      | 50.02%              | no data          |
|              | span selection      | no data                    | no data       | no data             |                  |
| Experiment 4 | both                | chinese-macbert-base       | 06:28:31      | 78.96%              | 76.37%           |

See report.pdf for detail settings.

official test accuracy:
- public: 0.76842, rank 164/202
- private: 0.78718, rank 146/202

note: other bert-base models can get better result (above 0.8) on test set.

## Test Script

The repository includes a helper script `test.sh` that performs an end-to-end environment and smoke test. It is a Bash script intended for Unix-like environments (Ubuntu, WSL, Git Bash). The script performs the following steps:

- Step 1 — Environment check
- Step 2 — Package check
- Step 3 — Run `download.sh`
- Step 4 — Run `run.sh`

Behavior and notes:

- The script prints the expected TA package versions and also attempts to create a `my_requirements.txt` (using `pipreqs`) and prints it.
- Logs are written to `download.log` and `run.log` in the same directory as `test.sh`.
- There are commented lines that can temporarily block network access using `iptables` (requires `sudo`) — these are not enabled by default.
- The script assumes `python3`, `bash` and common Unix utilities (`uname`, `free`, `df`, `awk`) are available.

How to run

```
./test.sh
```

Before running

- Edit the top of `test.sh` to set:

  - `CONTEXT_PATH` — path to the context JSON (relative or absolute)
  - `TEST_PATH` — path to the test JSON
  - `PREDICT_PATH` — path where `prediction.csv` will be written

- Make sure `download.sh` and `run.sh` are present and runnable in the same folder.
- If you run on a GPU machine, ensure `nvidia-smi` is available to show GPU info.

Files produced by the test

- `download.log` — stdout/stderr from `download.sh`
- `run.log` — stdout/stderr from `run.sh`

## Implementation Notes

- Paragraph selection adapts Hugging Face multiple-choice example (`run_swag_no_trainer.py`) for Chinese QA.
- Span selection adapts Hugging Face QA example (`run_qa_no_trainer.py`) and uses `postprocess_qa_predictions` logic.

## References

- Hugging Face examples used as references:
  - https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice
  - https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
- Pretrained models referenced in experiments:
  - https://huggingface.co/bert-base-chinese
  - https://huggingface.co/hfl/chinese-macbert-base