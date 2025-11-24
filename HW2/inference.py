import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import get_prompt, get_bnb_config, get_two_shots_prompt


def generate_outputs(model, tokenizer, data, batch_size=8, gen_max_new_tokens=256, two_shots=False, quick_test=False):
    # 準備 prompts
    if two_shots:
      prompts = [get_two_shots_prompt(item["instruction"]) for item in data]
    else:
      prompts = [get_prompt(item["instruction"]) for item in data]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    results = []
    # 批次處理
    for i in range(0, len(prompts), batch_size):
        input_ids = encodings["input_ids"][i:i+batch_size]
        attention_mask = encodings["attention_mask"][i:i+batch_size]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # decode 只取新生成部分
        decoded_batch = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )
        for j, decoded in enumerate(decoded_batch):
            item = {
                "id": data[i + j]["id"],
                "output": decoded.strip(),
            }
            if quick_test:
                item["prompt"] = prompts[i + j]
            results.append(item)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Base model path")
    parser.add_argument("--adapter_path", type=str, help="PEFT adapter path")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test on first 8 samples")
    parser.add_argument("--two_shots", action="store_true", help="Use two-shot prompts")
    args = parser.parse_args()

    # Load base model
    bnb_config = get_bnb_config()
    model_name = args.model_path if args.model_path else "Qwen/Qwen3-4B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = 'left' 

    # Load PEFT adapter
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # 如果 PyTorch 2.0+
    try:
        model = torch.compile(model)
    except Exception:
        pass

    # Load input
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Quick test
    if args.quick_test:
        data = [data[i] for i in [0, 7, 8, 11]]

    # Run inference
    results = generate_outputs(model, tokenizer, data, batch_size=args.batch_size, two_shots=args.two_shots, quick_test=args.quick_test)

    # Save output
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference finished. Results saved to {args.output_path}")
