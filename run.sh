# CUDA_VISIBLE_DEVICES=0,1,2,4 accelerate launch --config_file /home/xiazeyu21/sta_hw/4gpus.yaml /home/xiazeyu21/sta_hw/inference.py


# CUDA_VISIBLE_DEVICES=4,5,6,7 python inference_qwen.py

# CUDA_VISIBLE_DEVICES=0,1,2 python inference_llama.py

# CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed  inference_ds_new.py

CUDA_VISIBLE_DEVICES=4,5,6,7 python get_prob.py