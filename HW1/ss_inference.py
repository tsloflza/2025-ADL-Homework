#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('nvidia-smi')
# get_ipython().run_line_magic('pip', 'install datasets numpy torch accelerate pandas tqdm')


# In[ ]:


#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import csv
import collections
import json
import logging
import os
from types import SimpleNamespace
from typing import Optional

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
)


# In[ ]:


print(f"===== ss_inference.py =====")

# 讀取命令列參數
parser = argparse.ArgumentParser()
parser.add_argument("--test_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args_dict = parser.parse_args().__dict__

args = SimpleNamespace(
    test_file=args_dict["test_file"],
    ps_result="ps_result.csv",
    max_seq_length=512,
    pad_to_max_length=False,
    model_name_or_path="downloads/ss2_model",
    per_device_eval_batch_size=1,
    lr_scheduler_type=SchedulerType.LINEAR, 
    output_file=args_dict["output_file"],
    seed=1234,
    doc_stride=128,
    n_best_size=20,
    max_answer_length=30,
)

print(args)


# In[ ]:


def postprocess_qa_predictions(
    examples,
    features,
    predictions: tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        all_predictions[example["id"]] = predictions[0]["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise OSError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )

        print(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        print(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


# In[ ]:


def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    """
    Save results while prefixing metric names.

    Args:
        results: (:obj:`dict`):
            A dictionary of results.
        output_dir: (:obj:`str`):
            An output directory.
        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):
            An output file name.
        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):
            A metric name prefix.
    """
    # Prefix all keys with metric_key_prefix + '_'
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)


# In[ ]:


# ===== Accelerator =====
accelerator = Accelerator()

# Set seed for reproducibility
if args.seed is not None:
    set_seed(args.seed)


# In[ ]:


# === 載入 PS 結果 (paragraph selection 預測) ===
ps_df = pd.read_csv(args.ps_result)

# === 載入測試檔 (問題與 id) ===
with open(args.test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === 構建 span selection inference dataset ===
examples = []
for ex in test_data:
    qid = ex["id"]
    question = ex["question"]

    # 從 ps_result 找出對應的 paragraph (已經是文字)
    pred_para = ps_df.loc[ps_df["id"] == qid, "prediction"].values[0]

    examples.append({
        "id": qid,
        "question": question,
        "context": pred_para,
    })

# === 建立 Dataset ===
raw_datasets = datasets.DatasetDict({
    "test": datasets.Dataset.from_list(examples)
})

print(raw_datasets)
print(raw_datasets["test"][0])


# In[ ]:


# ========= Load pretrained model and tokenizer =========
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForQuestionAnswering.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
).to(device)
print(model.device)

# ========= Dataset column names =========
column_names = raw_datasets["test"].column_names
question_column_name = "question"
context_column_name = "context"

pad_on_right = tokenizer.padding_side == "right"
max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)


# In[ ]:


# ========= Preprocessing (Inference) =========
def prepare_validation_features(examples):
    # 去除開頭空白
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # tokenize question-context pair
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,   # 保留 offset_mapping
        padding="max_length" if args.pad_to_max_length else False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]

        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # 只保留 context 部分的 offset_mapping
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# Apply preprocessing
test_examples = raw_datasets["test"]
test_dataset = test_examples.map(
    prepare_validation_features,
    batched=True,
    remove_columns=column_names,
    desc="Tokenizing test dataset",
)

# 建立 dataset 給 model（去掉 offset_mapping）
test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])

# ========= DataLoader =========
if args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorWithPadding(tokenizer)

test_dataloader = DataLoader(
    test_dataset_for_model,
    collate_fn=data_collator,
    batch_size=args.per_device_eval_batch_size,
)

print(test_examples[0])


# In[ ]:


# ===== Inference =====
print("***** Running Inference *****")
print(f"  Num examples = {len(test_dataset)}")
print(f"  Batch size = {args.per_device_eval_batch_size}")

all_start_logits = []
all_end_logits = []
model.eval()

for step, batch in enumerate(tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process)):
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if not args.pad_to_max_length:
            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
        all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())


# ===== 整理 logits =====

# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

max_len = max(x.shape[1] for x in all_start_logits)
start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

# 清理暫存
del all_start_logits
del all_end_logits

# ===== 後處理，轉成文字答案 =====
outputs_numpy = (start_logits_concat, end_logits_concat)
predictions = postprocess_qa_predictions(
    examples=test_examples,
    features=test_dataset,
    predictions=outputs_numpy,
    version_2_with_negative=False,
    n_best_size=args.n_best_size,
    max_answer_length=args.max_answer_length,
    null_score_diff_threshold=0.0,
    output_dir=None,   # 不用輸出到中間檔
    prefix="test",
)


# In[ ]:


# ===== 存成 CSV =====
if accelerator.is_main_process:
    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for qid, ans in predictions.items():
            writer.writerow([qid, ans])

    print(f"✅ Saved predictions to {args.output_file}")

