from typing import List
import re
    
def parse_generated_answer(pred_ans: str) -> str:
    """Extract only the final assistant answer from the model output."""
    # Split at 'assistant' (if present)
    if "assistant" in pred_ans:
        pred_ans = pred_ans.split("assistant", 1)[-1]

    # Remove <think>...</think> sections if the model outputs reasoning traces
    pred_ans = re.sub(r"<think>.*?</think>", "", pred_ans, flags=re.DOTALL)

    # Clean leading/trailing quotes, whitespace, and markdown artifacts
    pred_ans = pred_ans.strip().strip('"').strip("'").strip()
    return pred_ans


# Method 1: basic
def get_inference_system_prompt():
    return "Answer the user concisely based on the context passages."

def get_inference_user_prompt(query, context_list):
    return f"Question: {query}\n\nContext:\n" + "\n\n".join(context_list)

# Method 2: CoT
def get_inference_system_prompt2():
    return "You are an assistant that answers questions using evidence from the given passages. Explain your reasoning briefly before giving the final answer."

def get_inference_user_prompt2(query, context_list):
    return f"Question: {query}\n\nRelevant Passages:\n" + "\n---\n".join(context_list) + "\n\nPlease reason step by step and conclude with a short final answer."

# method 3: Confidence-Based
def get_inference_system_prompt3():
    return "Provide an answer only if the context clearly supports it; otherwise, respond with 'CANNOTANSWER'."

def get_inference_user_prompt3(query, context_list):
    return f"Question: {query}\n\nSupporting Contexts:\n" + "\n\n".join(context_list)