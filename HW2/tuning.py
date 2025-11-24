import os
import json
import argparse
import random
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from utils import get_prompt, get_bnb_config


# ----------- Main -----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    # Load datasets
    with open(args.train_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=args.seed)
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply prompt formatting + tokenize
    def preprocess_function(examples):
        texts = [get_prompt(instr) + out for instr, out in zip(examples["instruction"], examples["output"])]
        model_inputs = tokenizer(
            texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        # Labels = input_ids (causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # Load model (QLoRA)
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # PEFT (LoRA)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
        eval_strategy="steps",
        save_strategy="steps",           # 每隔 logging_steps 保存一次
        save_total_limit=1,              # 只保留最佳模型
        load_best_model_at_end=True,     # 訓練完自動載入最佳權重
        metric_for_best_model="loss",    # 用 loss 來決定最佳
        greater_is_better=False,         # 越小越好

        bf16=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save final adapter
    model.save_pretrained(args.output_dir)
    print(f"Training finished. Model saved at {args.output_dir}")


if __name__ == "__main__":
    main()