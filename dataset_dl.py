import os
os.environ["HF_DATASETS_CACHE"] = "/gpfs/junlab/xiazeyu21/Datasets"

from datasets import load_dataset


dataset = load_dataset("HuggingFaceH4/MATH-500")  # 或 "socratic" 子集
