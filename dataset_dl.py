import os
os.environ["HF_DATASETS_CACHE"] = "/gpfs/junlab/xiazeyu21/Datasets"

from datasets import load_dataset


dataset = load_dataset("gsm8k", "main")  # 或 "socratic" 子集
