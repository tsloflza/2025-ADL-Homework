from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def get_two_shots_prompt(instruction: str) -> str:
    return f"""
你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。 \
USER: 我當時在三司，訪求太祖、仁宗的手書敕令沒有見到，然而人人能傳誦那些話，禁止私鹽的建議也最終被擱置。\n翻譯成文言文： ASSISTANT:餘時在三司，求訪兩朝墨敕不獲，然人人能誦其言，議亦竟寢。
USER: 議雖不從，天下鹹重其言。\n翻譯成白話文： ASSISTANT:他的建議雖然不被采納，但天下都很敬重他的話。
USER: {instruction} ASSISTANT:
            """

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig for 4bit quantization.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",      # 正常情況用 nf4 量化
        bnb_4bit_use_double_quant=True, # double quant 提升效率
        bnb_4bit_compute_dtype=torch.bfloat16, # 用 bfloat16 做運算
    )
