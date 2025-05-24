from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch, os

MODEL_PATH = "/gpfs/junlab/xiazeyu21/Models/R1-Distill-Qwen-1.5B"

# 1. 只在第一次手动修一次 config，之后就不必每次都 save_pretrained 了
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
cfg.num_key_value_heads = 2
cfg.multi_query = True
cfg.save_pretrained(MODEL_PATH)

# 2. 加载 tokenizer / model（注意 trust_remote_code=True）
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,        # ← 关键acc
    low_cpu_mem_usage=True,
)

model.eval()

# 3. 随便推两句试试
prompt = "Q: What is 25 plus 17?\nA:"
ids = tok([prompt], return_tensors="pt").to(model.device)

with torch.no_grad():
    outs = model.generate(**ids, max_new_tokens=16, do_sample=False)
print(tok.decode(outs[0], skip_special_tokens=True))
