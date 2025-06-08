import os, json, torch
from functools import partial
from pathlib import Path
from accelerate.utils import set_seed
from tqdm import tqdm
import torch.nn.functional as F 

os.environ["HF_DATASETS_CACHE"] = "/gpfs/junlab/xiazeyu21/Datasets"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_DIR = Path("/gpfs/junlab/xiazeyu21/Models/R1-Distill-Llama-8B")
BATCH_SIZE = 2
MAX_PROMPT_LEN = 512
MAX_NEW_TOKENS = 4096
SEED = 42
DATASET_SIZE = 500

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

# ---------- 0. 一劳永逸修补 config（只在首次运行需要） ----------
# cfg_path = MODEL_DIR / "config.json"
# cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
# updated = False
# if getattr(cfg, "num_key_value_heads", None) != 2:
#     cfg.num_key_value_heads = 2
#     cfg.multi_query = True
#     cfg.save_pretrained(MODEL_DIR)
#     updated = True
#     print("✔ 已修补 config.json 的 KV-heads=2")

# ---------- 1. 初始化 Accelerator ----------
accelerator = Accelerator()
accelerator.print(f"🐎 Accelerator initialised on {accelerator.device}")
set_seed(SEED)
# ---------- 2. 加载 tokenizer / model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.save_pretrained("dataset/llama_tokenizer")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",          # 让 HF 分卡，Accelerate 只处理 sync
    trust_remote_code=True,
)

# （可选）只在单卡场景尝试 compile
if torch.cuda.device_count() == 1:
    model = torch.compile(model)

# 将模型交给 accelerator；保持 tokenizer 在 CPU 即可
# model = accelerator.prepare(model)

# ---------- 3. 数据 ----------------------------------------------------------------
# ds = load_dataset("gsm8k", "main", split="test")
dataset_path = "/gpfs/junlab/xiazeyu21/Datasets/math-500"
ds = load_dataset(dataset_path)
ds = ds["test"]
ds = ds.select(range(min(len(ds), DATASET_SIZE)))
print(f'dataset size:{len(ds)}')

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
    # length = tok(prompts,
    #             padding="max_length",
    #             max_length=max_len,
    #             truncation=False,
    #             return_length=True)["length"]
    # print(f"max length:{max(length)}")
    return enc, prompts

loader = DataLoader(
    ds, batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(collate, tok=tokenizer, max_len=MAX_PROMPT_LEN)
)

loader = accelerator.prepare(loader)  # 让 dataloader 也能多卡同步


# ---------- 4. generation config ---------------------------------------------------
gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=MAX_NEW_TOKENS,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ---------- 5. 推理循环 ------------------------------------------------------------
model.eval()
ans_list      = []            # 答案文本
prob_chunks   = []            # 每个 batch 的 [bs, vocab] 概率

with torch.no_grad():
    for step, (batch_tok, raw_prompts) in tqdm(enumerate(loader), total=len(loader)):
        outputs = model.generate(
            **batch_tok,
            generation_config=gen_cfg,
            output_scores=False,
            return_dict_in_generate=True
        )

        # 1️⃣ 取最后一步 logits → prob
        logits_last = model(
            outputs.sequences[:, -1:],  # 只喂最后一个 token
            past_key_values=None,
        ).logits[:, -1, :]             # [B, vocab]
        probs_last = F.softmax(logits_last.float(), dim=-1)
        probs_gather = accelerator.gather_for_metrics(probs_last)  # 汇总各进程
        prob_chunks.append(probs_gather.cpu())               # 移回 CPU，节省显存

        # 2️⃣ 保存解码文本（和原来一样）
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        for full_text, prompt in zip(decoded, raw_prompts):
            ans_list.append(full_text[len(prompt):].strip())

# ---------- 6. 拼接得到 [dataset_size, vocab] --------------------------------------
all_step_probs = torch.cat(prob_chunks, dim=0)[:len(ds)]  # 万一多卡有 padding
print("probs shape =", all_step_probs.shape)
result = (ans_list, all_step_probs)


torch.save(result, f"/home/xiazeyu21/sta_hw/dataset/math500_{len(ds)}_llama.pt")
# ---------- 6. 打印前 10 条 ---------------------------------------------------------
# for i, (q, a) in enumerate(zip(ds["question"], ans_list[:10]), 1):
#     print(f"\nQ{i}: {q}\nA{i}: {a}\n" + "-" * 60)