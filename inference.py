#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GSM8K inference with distilled Qwen2 (1.5 B) using 🏎️ Accelerate.
"""
import os, json, torch
from functools import partial
from pathlib import Path
from accelerate.utils import set_seed
from tqdm import tqdm

os.environ["HF_DATASETS_CACHE"] = "/gpfs/junlab/xiazeyu21/Datasets"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_DIR = Path("/gpfs/junlab/xiazeyu21/Models/R1-Distill-Qwen-7B")
BATCH_SIZE = 32
MAX_PROMPT_LEN = 128
MAX_NEW_TOKENS = 128
SEED = 42

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
ds = load_dataset("gsm8k", "main", split="test").select(range(320))

def build_prompt(q: str) -> str:
    return f"Q: {q.strip()}\nA:"

def collate(batch, tok, max_len):
    prompts = [build_prompt(x["question"]) for x in batch]
    enc = tok(prompts,
              padding="max_length",
              max_length=max_len,
              truncation=True,
              return_tensors="pt")
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
all_out = []

with torch.no_grad():
    for step, (batch_tok, raw_prompts) in tqdm(enumerate(loader), total=len(loader)):
        # batch_tok 已在正确设备；无需 .to(model.device)
        outputs = model.generate(**batch_tok, generation_config=gen_cfg)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 剥掉 prompt，只留答案
        for full_text, prompt in zip(decoded, raw_prompts):
            ans = full_text[len(prompt):].strip()
            all_out.append(ans)
        # accelerator.print(f"✅ batch {step} done")

# ---------- 6. 打印前 10 条 ---------------------------------------------------------
for i, (q, a) in enumerate(zip(ds["question"], all_out[:10]), 1):
    print(f"\nQ{i}: {q}\nA{i}: {a}\n" + "-" * 60)
