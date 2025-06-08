#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combine alignment + grading for GSM8K predictions
Author: your-name-here
"""

import os, csv, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

import torch
from datasets import load_dataset
from openai import OpenAI

# ─── 配置区 ──────────────────────────────────────────────────────────────────────
DATASET_SIZE        = 500                       # 评测题目数量；None 表示整套
PRED_PATH           = "dataset/math500_500_llama.pt"    # torch.save 的 (ans_list, all_step_probs)
OUT_CSV             = f"{os.path.basename(PRED_PATH)}.csv" # 最终输出
MODEL_NAME          = "gpt-4o"                  # GPT 模型
TEMP                = 0
MAX_THREADS         = 20                        # GPT 并发线程
RATE_LIMIT_RPM      = 200                       # 你的帐户 60 秒请求上限
OPENAI_API_KEY      = "sk-1KdTGCbGuuBiQjsrPQCcT1RchAkcqmomtKxvPXrDsnxmimvB"
OPENAI_BASE_URL     = "https://pro.xiaoai.plus/v1"

SYSTEM_PROMPT = (
    "You are a strict grader. Given the TRUE numeric answer and the MODEL's "
    "answer, usually the number is at the end of the answer. "
    "Output only 'CORRECT' if they are equivalent, otherwise 'INCORRECT'. "
    "No explanation."
)
# ────────────────────────────────────────────────────────────────────────────────

_number_pat = re.compile(r"-?\d+(?:/\d+|\.\d+)?")

def extract_number(txt: str) -> str | None:
    """抓取文本中最后一个数字（整数/小数/分数）"""
    matches = _number_pat.findall(txt)
    return matches[-1].lstrip("+") if matches else None

def numeric_equal(a: str | None, b: str | None) -> bool:
    """支持整数、分数、小数比较"""
    if a is None or b is None:
        return False
    # 分数需要转为浮点
    def to_float(x):
        return eval(x) if "/" in x else float(x)
    try:
        return abs(to_float(a) - to_float(b)) < 1e-6
    except Exception:
        return False

def gpt_check(idx: int, question: str, true_answer: str, model_answer: str,
              client: OpenAI) -> Tuple[int, str]:
    """调用 GPT 判分，若出错返回 'INVALID' """
    user_prompt = (
        f"PROBLEM:\n{question}\n\n"
        f"TRUE ANSWER:\n{true_answer}\n\n"
        f"MODEL ANSWER:\n{model_answer}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMP,
        )
        out = resp.choices[0].message.content.strip().upper()
        return idx, "CORRECT" if out.startswith("CORRECT") else "INCORRECT"
    except Exception as e:
        print(f"[{idx}] GPT error: {e}")
        return idx, "INVALID"

def main() -> None:
    # 1️⃣ 载入数据与预测
    print("Loading dataset and predictions …")
    ans_list, _ = torch.load(PRED_PATH)
    dataset_path = "/gpfs/junlab/xiazeyu21/Datasets/math-500"
    ds = load_dataset(dataset_path)
    ds_full = ds["test"]
    ds = ds_full.select(range(DATASET_SIZE)) if DATASET_SIZE else ds_full

    if len(ds) != len(ans_list):
        print(f"Warning: len(ds)={len(ds)} vs len(pred)={len(ans_list)}, truncating …")
        min_len = min(len(ds), len(ans_list))
        ds       = ds.select(range(min_len))
        ans_list = ans_list[:min_len]

    n = len(ds)
    verdicts: List[str] = ["PENDING"] * n
    uncertain_idx: List[int] = []

    # 2️⃣ 先本地比较
    for i in range(n):
        true_num = extract_number(ds[i]["answer"])
        pred_num = extract_number(ans_list[i])
        if numeric_equal(true_num, pred_num):
            verdicts[i] = "CORRECT"
        else:
            uncertain_idx.append(i)

    print(f"Local numeric match CORRECT for {n - len(uncertain_idx)}/{n}, "
          f"{len(uncertain_idx)} left to GPT …")

    # 3️⃣ GPT 并发复核
    if uncertain_idx:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        with ThreadPoolExecutor(MAX_THREADS) as pool:
            futures = [
                pool.submit(
                    gpt_check,
                    idx=i,
                    question=ds[i]["problem"],
                    true_answer=ds[i]["answer"],
                    model_answer=ans_list[i],
                    client=client,
                )
                for i in uncertain_idx
            ]
            for f in as_completed(futures):
                idx, verdict = f.result()
                verdicts[idx] = verdict
                # 简单速率控制（粗糙，按 GPT 并发 & RATE_LIMIT_RPM 自行调）
                time.sleep(60.0 / RATE_LIMIT_RPM)

    # 4️⃣ 统计 & 写 CSV
    acc = sum(v == "CORRECT" for v in verdicts) / n
    print(f"Final accuracy = {acc:.2%}")

    print(f"Writing combined CSV to {OUT_CSV} …")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["idx", "problem", "reference_answer", "predicted_answer", "verdict"]
        )
        for i, v in enumerate(verdicts):
            writer.writerow([
                i,
                ds[i]["problem"].strip(),
                ds[i]["answer"].strip(),
                str(ans_list[i]).strip(),
                v,
            ])
    print("Done.")

if __name__ == "__main__":
    main()
