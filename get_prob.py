# -*- coding: utf-8 -*-
"""
给定 prompt 与 *已有* generation（存于 CSV），
计算模型在“倒数第二 token 位置”预测最后一个 token 的 logits。
支持多 GPU，使用 Accelerate device_map="auto"。
"""

import os, torch, json
import numpy as np, pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ---------- 配置 ----------
MODEL_DIR     = Path("/gpfs/junlab/xiazeyu21/Models/R1-Distill-Llama-8B")
CSV_PATH      = "gsm8k_7473_llama.pt.csv"      # ← 你的 CSV；需含 predicted_answer 列
GEN_COL       = "predicted_answer"
DATASET_PATH  = "/gpfs/junlab/xiazeyu21/Datasets/gsm8k"
DATASET_SIZE  = 7473
BATCH_SIZE    = 5
MAX_PROMPT_LEN= 512                         # prompt 截断上限
OUT_FILE      = f"/home/xiazeyu21/sta_hw/dataset/gsm8k_logits_llama_{DATASET_SIZE}.pt"
SEED          = 42
# -----------------------------------------

# 1. accelerator 初始化
accelerator = Accelerator()
set_seed(SEED)
accelerator.print(f"Accelerator device: {accelerator.device}")

# 2. tokenizer / model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# 3. 数据加载（dataset + csv generations）
ds = load_dataset(DATASET_PATH)["train"]
if DATASET_SIZE:
    ds = ds.select(range(min(len(ds), DATASET_SIZE)))

gen_df = pd.read_csv(CSV_PATH)
assert GEN_COL in gen_df.columns, f"CSV 中缺列 {GEN_COL}"
assert len(gen_df) >= len(ds),    "CSV 行数不足"

questions   = ds["question"]
generations = gen_df[GEN_COL].iloc[: len(questions)].tolist()

def build_prompt(q: str) -> str:
    return (
        "You are a helpful and harmless assistant.\n"
        "You should think step-by-step.\n"
        f"Question: {q.strip()}\n\n"
        "Here is your answer:\n"
    )

def collate(batch_idx, tok, max_len):
    # batch_idx 是索引列表
    full_texts, prompts = [], []
    for idx in batch_idx:
        prompt = build_prompt(questions[idx])
        gen    = generations[idx]
        full   = prompt + gen
        full_texts.append(full)
        prompts.append(prompt)
    enc = tok(full_texts,
              padding=True,
              truncation=False,   # 不截 generation；如需可设置 max_length
              return_tensors="pt")
    # length = tok(full_texts,
    #           padding=True,
    #           truncation=False,   # 不截 generation；如需可设置 max_length
    #           return_tensors="pt",
    #           return_length=True)["length"]
    # print(f"max length:{max(length)}")
    return enc, prompts

indices = list(range(len(ds)))
loader  = DataLoader(
    indices, batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(collate, tok=tokenizer, max_len=MAX_PROMPT_LEN)
)
loader = accelerator.prepare(loader)

# 4. 推理：取最后 logits
ans_texts, prob_chunks = [], []

with torch.no_grad():
    for enc, prompts in tqdm(loader, total=len(loader)):
        # a) 把张量送到模型首层设备（Accelerate 自动广播）
        for k in enc:
            enc[k] = enc[k].to(model.device)

        # b) 去掉最后 token
        seq_len = enc["attention_mask"].sum(1)   # [B]
        seq_trim, attn_trim = [], []
        for ids, attn, L in zip(enc["input_ids"], enc["attention_mask"], seq_len):
            seq_trim.append(ids[: L-1])
            attn_trim.append(attn[: L-1])

        enc_trim = tokenizer.pad(
            {"input_ids": seq_trim},
            padding=True, return_tensors="pt"
        )
        attn_pad = torch.zeros_like(enc_trim["input_ids"])
        for r, a in enumerate(attn_trim):
            attn_pad[r, : a.size(0)] = 1

        enc_trim["attention_mask"] = attn_pad.to(model.device)
        enc_trim["input_ids"]      = enc_trim["input_ids"].to(model.device)

        # c) 前向，取最后 logits
        out_logits = model(**enc_trim).logits      # [B, L-1, V]
        last_logits = out_logits[:, -1, :]         # [B, V]
        probs = F.softmax(last_logits.float(), dim=-1)

        # d) 汇总 & 保存
        probs_g = accelerator.gather_for_metrics(probs)
        if accelerator.is_main_process:
            prob_chunks.append(probs_g.cpu())

        # e) 保存回答文本（generation 已知）
        if accelerator.is_main_process:
            full_texts = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)
            for full, prompt in zip(full_texts, prompts):
                ans_texts.append(full[len(prompt):].strip())

if accelerator.is_main_process:
    all_probs = torch.cat(prob_chunks, 0)[: len(ds)]   # [N, vocab]
    torch.save((ans_texts, all_probs), OUT_FILE)
    print("Saved:", OUT_FILE, all_probs.shape)
