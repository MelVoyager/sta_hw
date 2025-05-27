import torch
ans_list, all_step_probs = torch.load("dataset/gsm8k_7473_qwen.pt")
# result = torch.load("gsm8k_qwen2_7b_tp4_logits152k.pt")
# all_step_probs, ans_list = result["logits"], result["answers"]
all_step_probs[:100].to('cuda')
print(all_step_probs.shape)
probs_sum = torch.sum(all_step_probs, dim=1)
print(probs_sum.shape)
print(probs_sum)

# for i in range(len(ans_list[:10])):
#     print(f"{i}:")
    # print(ans_list[i])