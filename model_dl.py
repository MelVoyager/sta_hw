from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    local_dir="/gpfs/junlab/xiazeyu21/Models/R1-Distill-Qwen-14B",
    local_dir_use_symlinks=False,
    resume_download=True
)
