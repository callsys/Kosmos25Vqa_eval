import os
import copy
import json

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

class rvlcdip_dataset(Dataset):
    def load_data(self, cur_path):
        def load_path_and_label(path):
            data = []
            with open(path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    if len(line) == 0: continue
                    fields = line.split()
                    if len(fields) != 2: continue
                    ocr_path, label = fields
                    ocr_path = os.path.join(self.args.data_dir, 'ocr', ocr_path + '.ocr.json')
                    label = int(label)
                    if os.path.exists(ocr_path) == False: continue
                    data.append([ocr_path, label])
            return data

        def load_ocr(paths):
            def bbox_norm(bbox, width, height):
                def clip(min_num, num, max_num):
                    return min(max(num, min_num), max_num)

                x = [bbox[i] for i in range(0, len(bbox), 2)]
                y = [bbox[i] for i in range(1, len(bbox), 2)]

                x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
                x0 = clip(0, int((x0 / width) * 1000), 1000)
                y0 = clip(0, int((y0 / height) * 1000), 1000)
                x1 = clip(0, int((x1 / width) * 1000), 1000)
                y1 = clip(0, int((y1 / height) * 1000), 1000)

                return [x0, y0, x1, y1]

            whole_data = {
                "input_ids": [],
                'segment_ids': [],
                'labels': [],
                'bbox': [],
                'position_ids': [],
                "attention_mask": []
            }
            for i in tqdm(range(len(paths)), desc='loading data'):
                if i == 10000: break
                try:
                    with open(paths[i][0], 'r', encoding='utf-8') as fr:
                        data = json.load(fr)['analyzeResult']['readResults'][0]
                        width, height = data['width'], data['height']

                        cur_doc = {
                            "input_ids": [],
                            'segment_ids': [],
                            'bbox': [],
                            'position_ids': [],
                            'labels': None,
                        }

                        for index, line in enumerate(data['lines']):
                            cur_text = line['text'].strip()
                            if len(cur_text) == 0: continue
                            cur_bbox = bbox_norm(line['boundingBox'], width, height)

                            tokenized_line = self.tokenizer(cur_text, add_special_tokens=False, return_attention_mask=False)
                            tokenized_line['bbox'] = [cur_bbox] * len(tokenized_line['input_ids'])
                            tokenized_line['segment_ids'] = [1 + index for j in range(len(tokenized_line['input_ids']))]
                            tokenized_line['position_ids'] = [2 + j for j in range(len(tokenized_line['input_ids']))]

                            cur_doc['input_ids'] += tokenized_line['input_ids']
                            cur_doc['segment_ids'] += tokenized_line['segment_ids']
                            cur_doc['labels'] = paths[i][1]
                            cur_doc['bbox'] += tokenized_line['bbox']
                            cur_doc['position_ids'] += tokenized_line['position_ids']

                        cur_doc['input_ids'] = [0] + cur_doc['input_ids'] + [2]
                        cur_doc['segment_ids'] = [0] + cur_doc['segment_ids'] + [cur_doc['segment_ids'][-1] + 1]
                        cur_doc['bbox'] = [[0, 0, 0, 0]] + cur_doc['bbox'] + [[1000, 1000, 1000, 1000]]
                        cur_doc['position_ids'] = [2] + cur_doc['position_ids'] + [2]

                        assert len(cur_doc['input_ids']) == len(cur_doc['segment_ids']) == len(cur_doc['bbox']) == len(
                            cur_doc['position_ids'])
                        cur_doc['attention_mask'] = [1] * len(cur_doc['input_ids'])


                        whole_data['input_ids'].append(cur_doc['input_ids'][:self.args.max_length])
                        whole_data['segment_ids'].append(cur_doc['segment_ids'][:self.args.max_length])
                        whole_data['labels'].append(cur_doc['labels'])
                        whole_data['bbox'].append(cur_doc['bbox'][:self.args.max_length])
                        whole_data['position_ids'].append(cur_doc['position_ids'][:self.args.max_length])
                        whole_data['attention_mask'].append(cur_doc['attention_mask'][:self.args.max_length])

                        whole_data['input_ids'][-1][-1] = 2
                        whole_data['segment_ids'][-1][-1] = whole_data['segment_ids'][-1][-2] + 1
                        whole_data['bbox'][-1][-1] = [1000, 1000, 1000, 1000]
                        whole_data['position_ids'][-1][-1] = 2
                except:
                    continue

            return whole_data


        path_and_label = load_path_and_label(cur_path)
        return load_ocr(path_and_label)

    def __init__(
            self,
            data_args,
            tokenizer,
            mode,
            label_num=16,
    ):
        self.args = data_args
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = self.args.max_length

        self.sep_token_bbox = [1000, 1000, 1000, 1000]
        self.pad_token_bbox = [0, 0, 0, 0]

        self.label_num = label_num
        self.mode = mode

        cur_path = os.path.join(data_args.data_dir, "labels", "{}.txt".format(mode))
        assert os.path.exists(cur_path)
        self.feature = self.load_data(cur_path)

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        attention_mask = self.feature["attention_mask"][index]
        labels = self.feature["labels"][index]
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        assert len(input_ids) == len(attention_mask) == len(bbox) == len(segment_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "bbox": bbox,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
        }


def pre_calc_rel_mat(position_ids, segment_ids):
    rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

    valid_span = torch.zeros_like(rel_pos_mat, device=rel_pos_mat.device, dtype=torch.bool)
    for i in range(rel_pos_mat.shape[0]):
        for j in range(rel_pos_mat.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    arange_list = torch.arange(0, position_ids.shape[1], step=1, device=position_ids.device).expand_as(position_ids)
    full_rel = arange_list.unsqueeze(-2) - arange_list.unsqueeze(-1)
    MAX_POS_DIS = 512
    MAX_NEG_DIS = -512

    rel_pos_mat[(full_rel > 0) & (valid_span == False)] = MAX_POS_DIS
    rel_pos_mat[(full_rel < 0) & (valid_span == False)] = MAX_NEG_DIS

    return rel_pos_mat

class rvlcdip_collator():
    def __init__(
            self,
            token_pad_id=0,
            bbox_pad_id=[0, 0, 0, 0],
            attention_mask_pad_id=0,
            token_type_pad_id=0,
            label_ignore=-100,
    ):
        self.token_pad_id = token_pad_id
        self.bbox_pad_id = bbox_pad_id
        self.attention_mask_pad_id = attention_mask_pad_id
        self.token_type_pad_id = token_type_pad_id
        self.label_ignore = label_ignore

    def __call__(self, batch):
        ret = {key: [copy.deepcopy(sample[key]) for sample in batch] for key, _ in batch[0].items()}
        max_length = max([len(input_ids) for input_ids in ret['input_ids']])

        pad_item = {
            'input_ids': self.token_pad_id,
            'bbox': self.bbox_pad_id,
            'attention_mask': self.attention_mask_pad_id,
            'token_type_ids': self.token_type_pad_id,
            "position_ids": 2,
        }

        for key, vals in ret.items():
            if key not in  ['segment_ids', 'labels']:
                for i in range(len(vals)):
                    ret[key][i] = vals[i] + [pad_item[key]] * (max_length - len(vals[i]))
                ret[key] = torch.tensor(ret[key], dtype=torch.long)
            else:
                if key in ['segment_ids']:
                    for i in range(len(vals)):
                        ret[key][i] = vals[i] + [vals[i][-1] + 1] * (max_length - len(vals[i]))
                    ret[key] = torch.tensor(ret[key], dtype=torch.long)
        ret['labels'] = torch.tensor(ret['labels'], dtype=torch.long)

        # pre calc the rel mat to reduce the training time
        rel_pos_mat = pre_calc_rel_mat(
            position_ids = ret['position_ids'],
            segment_ids = ret['segment_ids']
        )
        ret['rel_pos_mat'] = rel_pos_mat
        del ret['segment_ids']

        return ret