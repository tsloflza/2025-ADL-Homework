#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('nvidia-smi')
# get_ipython().run_line_magic('pip', 'install datasets torch tqdm')


# In[ ]:


#!/usr/bin/env python
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import csv
import json
from types import SimpleNamespace

import datasets
import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
)


# In[ ]:


print(f"===== ps_inference.py =====")

# 讀取命令列參數
parser = argparse.ArgumentParser()
parser.add_argument("--context_file", type=str, required=True)
parser.add_argument("--test_file", type=str, required=True)
args_dict = parser.parse_args().__dict__

args = SimpleNamespace(
    test_file=args_dict["test_file"],
    context_file=args_dict["context_file"],
    max_seq_length=512,
    pad_to_max_length=False,
    model_name_or_path="downloads/ps2_model",
    per_device_eval_batch_size=1,
    output_path="ps_result.csv",
    seed=1234,
)

print(args)


# In[ ]:


# Set the seed now.
if args.seed is not None:
    set_seed(args.seed)


# In[ ]:


# Get the datasets

with open(args.context_file, "r", encoding="utf-8") as f:
    contexts = json.load(f)

# Inference dataset loader (without label)
def load_paragraph_selection_test(file_path, contexts):
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    data = {
        "id": [],
        "question": [],
        "paragraphs": [],
    }

    for ex in examples:
        qid = ex["id"]
        question = ex["question"]
        para_ids = ex["paragraphs"]

        para_texts = [contexts[pid] for pid in para_ids]

        data["id"].append(qid)
        data["question"].append(question)
        data["paragraphs"].append(para_texts)

    return datasets.Dataset.from_dict(data)

# load test split
dataset_splits = {}
if args.test_file is not None:
    dataset_splits["test"] = load_paragraph_selection_test(args.test_file, contexts)

raw_datasets = datasets.DatasetDict(dataset_splits)

print(raw_datasets)
print(raw_datasets["test"][0])


# In[ ]:


# 1. 載入 config
config = AutoConfig.from_pretrained(args.model_name_or_path)

# 2. 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# 3. 載入 model (只做推論，不需要 from_tf)
model = AutoModelForMultipleChoice.from_pretrained(
    args.model_name_or_path,
    config=config
)

# 4. 調整 embedding 大小（避免 tokenizer 新增字典造成 index error）
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# 5. padding 方式 (inference 通常會用 dynamic padding 比較快)
padding = "max_length" if args.pad_to_max_length else False


# In[ ]:


# --- Preprocessing (for inference, no labels) ---
def preprocess_function(examples):
    questions = examples["question"]             # list[str]
    paragraphs_list = examples["paragraphs"]     # list[list[str]]

    first_sentences = []
    second_sentences = []

    for q, paras in zip(questions, paragraphs_list):
        assert len(paras) == 4, f"每題應該要有 4 個選項，但得到 {len(paras)}"
        first_sentences.extend([q] * 4)
        second_sentences.extend(paras)

    # Tokenize (flat)
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=args.max_seq_length,
        padding="max_length" if args.pad_to_max_length else False,
        truncation=True,
    )

    # reshape → [batch_size, num_choices, seq_len]
    result = {k: [v[i:i + 4] for i in range(0, len(v), 4)]
              for k, v in tokenized_examples.items()}

    # 保留 id 跟 paragraphs，方便後續輸出
    result["id"] = examples["id"]
    result["paragraphs"] = examples["paragraphs"]

    return result


# --- Dataset preprocessing ---
processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["test"].column_names
)
test_dataset = processed_datasets["test"]


# --- Collator (for inference only) ---
def inference_collator(features):
    ids = [f.pop("id") for f in features]
    paras = [f.pop("paragraphs") for f in features]

    # 展平成 flat list
    flat_features = []
    for f in features:
        for i in range(len(f["input_ids"])):  # num_choices (4)
            flat_features.append({k: f[k][i] for k in f.keys()})

    # padding → 再 reshape 回 [batch, num_choices, seq_len]
    batch = tokenizer.pad(
        flat_features,
        padding=True,
        return_tensors="pt"
    )

    batch_size = len(features)
    num_choices = len(features[0]["input_ids"])
    for k in batch.keys():
        batch[k] = batch[k].view(batch_size, num_choices, -1)

    batch["id"] = ids
    batch["paragraphs"] = paras
    return batch


# --- DataLoader ---
test_dataloader = DataLoader(
    test_dataset,
    collate_fn=inference_collator,
    batch_size=args.per_device_eval_batch_size
)


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# In[ ]:


# ===== Inference =====
ps_predictions = []

for step, batch in enumerate(tqdm(test_dataloader, desc="Running Inference")):
    inputs = {k: v.to(device) for k, v in batch.items() if k not in ["id", "paragraphs"]}

    with torch.no_grad():
        outputs = model(**inputs)

    pred_choices = outputs.logits.argmax(dim=-1).cpu().numpy()

    for i, choice in enumerate(pred_choices):
        qid = batch["id"][i]
        pred_para = batch["paragraphs"][i][choice]
        ps_predictions.append({
            "id": qid,
            "prediction": pred_para
        })


# In[ ]:


# ===== Save Inference Results =====
if args.output_path is not None:
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(ps_predictions)

    print(f"✅ Saved predictions to {args.output_path}")

