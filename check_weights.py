import torch

def get_vision_weights(weight):
    return [(key, value) for key, value in weight['model'].items() if "img_model" in key]

kosmos25_org = "/home/yuzhongzhao/zyz/ckpts/checkpoint_1_210000.pt"

kosmos25_vqa = "/home/yuzhongzhao/zyz/ckpts/checkpoint_1_1800.pt"
kosmos25_vqa2 = "/home/yuzhongzhao/zyz/ckpts/checkpoint_1_200.pt"

dct_org = torch.load(kosmos25_org)
dct_org_vision = get_vision_weights(dct_org)
dct_vqa = torch.load(kosmos25_vqa)
dct_vqa_vision = get_vision_weights(dct_vqa)
dct_vqa2 = torch.load(kosmos25_vqa2)
dct_vqa2_vision = get_vision_weights(dct_vqa2)

print(0)