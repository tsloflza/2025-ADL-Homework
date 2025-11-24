# NTU ADL 2025 — HW3

## Overview

This repository implements a Retrieval-Augmented Generation (RAG) system. The pipeline includes:

- a dense bi-encoder retriever (fine-tunable),
- a cross-encoder reranker (fine-tunable), and
- a language model-based generator for answer synthesis.

The code supports training, building a passage index (FAISS + SQLite), offline inference (retriever → reranker → LLM), and a small RL loop to tune the number of retrieved passages sent to the generator.

## Repository structure

- `download.sh`           — Script to download pretrained / fine-tuned models used in experiments.
- `report.pdf`            — Experiment report and detailed settings.
- `utils.py`              — Prompt templates and generated-answer parsing helpers.
- `inference_batch.py`    — Baseline RAG inference pipeline (retriever → reranker → generator) and grading function.
- `inference_batch_new.py`— Modified inference script with result saving and analysis utilities.
- `retriever.py`          — Retriever fine-tuning script.
- `reranker.py`           — Reranker fine-tuning script and evaluation utilities.
- `save_embeddings.py`    — Build corpus, create FAISS index and SQLite passage store.
- `rl.py`                 — Small RL setup (A2C-like) to tune Top-M selection for generation.
- `requirements.txt`      — Python dependencies used for experiments.
- `test.sh`               — Environment and smoke test wrapper.

## Quick start

1. Install dependencies
```
pip install -r requirements.txt
```

2. Download model checkpoints (if not already available)
```
bash ./download.sh
```

3. Build corpus and FAISS index (example)
```
python save_embeddings.py --retriever_model_path ./models/retriever --data_folder ./data --build_db
```
This creates the passage store and FAISS index under the configured output folder (default `./vector_database`).

4. Fine-tune the retriever
```
python retriever.py
```

5. Fine-tune the reranker
```
python reranker.py
```

6. Run inference
```
python inference_batch.py --test_data_path ./data/test.json --retriever_model_path ./models/retriever --reranker_model_path ./models/reranker
```
Or use the enhanced script which saves per-query outputs and analysis:
```
python inference_batch_new.py --retriever_model_path ./models/retriever --reranker_model_path ./models/reranker --test_data_path ./data/test.json
```

7. (Optional) Run RL-based Top-M tuning
```
python rl.py
```

## Evaluation & outputs

- Retrieval metrics (Recall@K) and reranker metrics (MRR@K) are computed by the inference scripts.
- `inference_batch_new.py` saves per-query results and an analysis JSON under `./output/` by default.
- The LLM generator outputs are parsed by helpers in `utils.py` for automatic evaluation.

### Reported results (from experiments)

| Metric | Value |
|--------|-------|
| recall@10 (retriever) | 0.8256 |
| mrr@10 (reranker)      | 0.7302 |
| LLM CosSim (gen vs gold)| 0.3660 |
| LLM CosSim (with RL)    | 0.3910 |

## Notes & implementation details

- Prompts and parsing: centralised in `utils.py` (`get_inference_system_prompt`, `get_inference_user_prompt`, `parse_generated_answer`).
- Retriever: uses a SentenceTransformers-style bi-encoder and training with contrastive/MNR-style loss (see `retriever.py`).
- Reranker: cross-encoder training and a custom evaluator that interfaces with the FAISS retrieval backend (see `reranker.py`).
- Inference: scripts load the retriever, reranker, and an LLM via `transformers` (tokenizer + model) and run batched generation. Batching and memory parameters are configurable in the scripts.
- RL: the `rl.py` module contains a small environment and policy that chooses Top-M; results showed a modest improvement in generated-answer similarity.

## Requirements

See `requirements.txt` for package versions used in experiments. Typical environment:
- Python 3.8+
- PyTorch matching local CUDA (or CPU-only)
- `transformers`, `sentence-transformers`, `datasets`, `faiss-cpu` (or `faiss-gpu`), `accelerate`, `peft` (if fine-tuning with LoRA), and other utilities.

## References
* Large Language Models
  * Wang, L., Yang, N., Wang, X., Joty, S., & Lin, J. (2022). Text Embeddings by Reranking. arXiv preprint arXiv:2212.03534. [https://arxiv.org/abs/2212.03534](https://arxiv.org/abs/2212.03534)
  * Reimers, Nils, and Iryna Gurevych. (2019) "Sentence-bert: Sentence embeddings using siamese bert-networks." arXiv preprint arXiv:1908.10084. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

* Finetuning & RL
  * Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP (1). 2020.
  * Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. PmLR, 2016.