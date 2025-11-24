# NTU ADL 2025 Homework

## HW1

### Task : Chinese Extractive Question Answering

The goal is to answer a given question based on a large collection of paragraphs written in Chinese.(always 4 candidate paragraphs in our cases)

### Implementation

The task is split into two stages:

- Paragraph Selection (PS): choose the most relevant paragraph (4 candidates) for a given question.
- Span Selection (SS): extract the exact answer span from the selected paragraph.

### Bonus

Single-model QA: Implemented by concatenating the four candidate paragraphs into one input and adjusting each answer's start index by the paragraph offset.

## HW2

### Task : Bidirectional translation between classical Chinese and modern Chinese

Given classical/modern Chinese sentence and instruction, translation the sentence to modern/classical Chinese

### Implementation

* **Fine-tuning:** A causal language model is trained using QLoRA + LoRA. The pipeline applies k-bit quantization and uses custom prompt formatting. The resulting adapters are saved to the designated output directory for later inference.

* **Data format:** Training and evaluation data are stored as JSON or JSON-lines, each containing paired instruction/input–target examples. Both translation directions are included in the same dataset so the model can learn classical→modern and modern→classical conversion within one adapter.

### Bonus

Do the same task on Llama3-Taiwan (8B).

## HW3

### Task : RAG system

1. Fine-tune retriever
2. Fine-tune reranker
3. Design prompt to optimize the generation performance

### Implementation

* **Retriever:** Trains a dense bi-encoder using contrastive or MNR-style objectives, then generates and stores passage embeddings in a FAISS index alongside a lightweight passage database.

* **Reranker:** Fine-tunes a cross-encoder to re-score the top-k retrieved passages, optimizing ranking quality and integrating seamlessly with the FAISS-based retrieval stage.

* **Generator & Prompts:** Designs system and user prompts for the LLM generator and parses model outputs using helper utilities. Similarity-based automatic metrics are used for evaluation.

### Bonus
RAG with RL for deciding the TOP_M (how many retrieved passages to use for LLM generation)