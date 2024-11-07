import os
import tqdm
import json

vqa_data = json.load(open("/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/vqa/test.json"))
data_dir = "/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/"
save_path = "./qids.txt"

samples = vqa_data[0]['source'][:200]
qids = []

for idx, sample in enumerate(tqdm.tqdm(samples)):

    file_path = os.path.join(data_dir, sample)
    if not os.path.exists(file_path):
        print('| file {} not exists'.format(file_path), flush=True)

    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    line = lines[0]

    assert len(line.strip()) != 0
    doc_str = json.loads(line.strip())
    qids.append(str(doc_str["extra_info"]["questionId"]) + "\n")
    # print(qids)

with open(save_path, "w") as fw:
    fw.writelines(qids)

print(0)