import torch
ans_list, all_step_probs = torch.load("dataset/math500_logits_qwen_500.pt")
# result = torch.load("gsm8k_qwen2_7b_tp4_logits152k.pt")
# all_step_probs, ans_list = result["logits"], result["answers"]
all_step_probs[:00].to('cuda')
print(all_step_probs.shape)
a = all_step_probs

for i in range(10):
    result = []
    for j in range(10):
        result.append(torch.equal(a[i], a[j]))
    print(result)
        
probs_sum = torch.sum(all_step_probs, dim=1)
print(probs_sum.shape)
print(probs_sum)

# for i in range(len(ans_list[:10])):
#     print(f"{i}:")
    # print(ans_list[i])