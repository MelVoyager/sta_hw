from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch, os
from datasets import load_dataset
from pathlib import Path
# MODEL_PATH = "/gpfs/junlab/xiazeyu21/Models/R1-Distill-Qwen-7B"

# # 1. 只在第一次手动修一次 config，之后就不必每次都 save_pretrained 了
# cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
# cfg.num_key_value_heads = 2
# cfg.multi_query = True
# cfg.save_pretrained(MODEL_PATH)

# # 2. 加载 tokenizer / model（注意 trust_remote_code=True）
# tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
# tok.pad_token_id = tok.eos_token_id

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,        # ← 关键acc
#     low_cpu_mem_usage=True,
# )

# model.eval()

# # 3. 随便推两句试试
# prompt = "Q: What is 25 plus 17?\nA:"
# ids = tok([prompt], return_tensors="pt").to(model.device)

# with torch.no_grad():
#     outs = model.generate(**ids, max_new_tokens=16, do_sample=False)
# print(tok.decode(outs[0], skip_special_tokens=True))
def build_prompt(q: str) -> str:
    return f"""You are a helpful and harmless assistant. You should think step-by-step.
Question: {q.strip()}\n
Here is your answer:\n"""

def collate(batch, tok, max_len):
    prompts = [build_prompt(x["problem"]) for x in batch]
    enc = tok(prompts,
              padding="max_length",
              max_length=max_len,
              truncation=True,
              return_tensors="pt")
    length = tok(prompts,
                padding="max_length",
                max_length=max_len,
                truncation=False,
                return_length=True)["length"]
    
    print(f"max length:{max(length)}")
    count = sum(l > max_len for l in length)
    print(f"Number of prompts with length > {max_len}: {count}")
    return enc, prompts


DATASET_SIZE = 600
os.environ["HF_DATASETS_CACHE"] = "/gpfs/junlab/xiazeyu21/Datasets"
MODEL_DIR = Path("/gpfs/junlab/xiazeyu21/Models/R1-Distill-Llama-8B")
dataset_path = "/gpfs/junlab/xiazeyu21/Datasets/math-500"
ds = load_dataset(dataset_path)
ds = ds["test"]
ds = ds.select(range(min(len(ds), DATASET_SIZE)))
print(f'dataset size:{len(ds)}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id


collate(ds, tokenizer, 512)