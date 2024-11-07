import os
import copy
import json

import torch
from torch.utils.data.dataset import Dataset

id2labels = \
{
    0: "O",
    1: 'B-HEADER',
    2: 'I-HEADER',
    3: 'B-QUESTION',
    4: 'I-QUESTION',
    5: 'B-ANSWER',
    6: 'I-ANSWER',
}

label2ids = \
{
    "O":0,
    'B-HEADER':1,
    'I-HEADER':2,
    'B-QUESTION':3,
    'I-QUESTION':4,
    'B-ANSWER':5,
    'I-ANSWER':6,
}


class funsd_dataset(Dataset):
    def load_data(
            self,
            data_file,
    ):
        def get_segment_ids(bboxs):
            segment_ids = []
            for i in range(len(bboxs)):
                if i == 0:
                    segment_ids.append(0)
                else:
                    if bboxs[i - 1] == bboxs[i]:
                        segment_ids.append(segment_ids[-1])
                    else:
                        segment_ids.append(segment_ids[-1] + 1)
            return segment_ids

        def get_position_ids(segment_ids):
            position_ids = []
            for i in range(len(segment_ids)):
                if i == 0:
                    position_ids.append(2)
                else:
                    if segment_ids[i] == segment_ids[i - 1]:
                        position_ids.append(position_ids[-1] + 1)
                    else:
                        position_ids.append(2)
            return position_ids

        tokenized_inputs = self.tokenizer(
            data_file['words'],
            # boxes=data_file['bboxes'],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        total_bboxes, total_segment_ids, total_labels_ids, total_language_ids, total_position_ids = [], [], [], [], []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            label_ids = data_file["ner_tags"][org_batch_index]
            bboxs = data_file['bboxes'][org_batch_index]

            previous_word_idx = None

            cur_new_labels_ids, cur_new_bbox_inputs = [], []
            for word_idx in word_ids:
                if word_idx is None:
                    cur_new_labels_ids.append(-100)
                    cur_new_bbox_inputs.append(self.pad_token_bbox)
                elif word_idx != previous_word_idx:
                    cur_new_labels_ids.append(label_ids[word_idx])
                    cur_new_bbox_inputs.append(bboxs[word_idx])
                else:
                    cur_new_labels_ids.append(-100)
                    cur_new_bbox_inputs.append(bboxs[word_idx])
                previous_word_idx = word_idx
            cur_new_bbox_inputs[-1] = self.sep_token_bbox

            assert len(cur_new_labels_ids) == len(cur_new_bbox_inputs) == len(word_ids)
            segment_ids = get_segment_ids(cur_new_bbox_inputs) # identify the text blocks
            cur_new_position_ids = get_position_ids(segment_ids) # identify the token position in text blocks

            assert len(segment_ids) == len(cur_new_bbox_inputs) == len(cur_new_labels_ids) == len(
                cur_new_position_ids)

            total_bboxes.append(cur_new_bbox_inputs)
            total_segment_ids.append(segment_ids)
            total_labels_ids.append(cur_new_labels_ids)
            total_position_ids.append(cur_new_position_ids)

        tokenized_inputs["segment_ids"] = total_segment_ids
        tokenized_inputs["labels"] = total_labels_ids
        tokenized_inputs["bbox"] = total_bboxes
        tokenized_inputs["language_ids"] = total_language_ids
        tokenized_inputs["position_ids"] = total_position_ids

        return tokenized_inputs


    def __init__(
            self,
            data_args,
            tokenizer,
            mode,
    ):
        self.args = data_args
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = self.args.max_length

        self.sep_token_bbox = [1000, 1000, 1000, 1000]
        self.pad_token_bbox = [0, 0, 0, 0]

        self.label2ids = label2ids
        self.id2labels = id2labels
        assert len(self.label2ids) == len(self.id2labels)

        self.mode = mode

        cur_path = os.path.join(data_args.data_dir, "{}.json".format('train' if mode == 'train' else 'test'))
        data_files = json.load(open(cur_path, 'r'))

        self.feature = self.load_data(
            data_file=data_files,
        )

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        attention_mask = self.feature["attention_mask"][index]
        labels = self.feature["labels"][index]
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        assert len(input_ids) == len(attention_mask) == len(labels) == len(bbox) == len(segment_ids)

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

class funsd_collator():
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
            'labels': self.label_ignore,
            "position_ids": 2,
        }

        for key, vals in ret.items():
            if key not in  ['segment_ids']:
                for i in range(len(vals)):
                    ret[key][i] = vals[i] + [pad_item[key]] * (max_length - len(vals[i]))
                ret[key] = torch.tensor(ret[key], dtype=torch.long)
            else:
                if key in ['segment_ids']:
                    for i in range(len(vals)):
                        ret[key][i] = vals[i] + [vals[i][-1] + 1] * (max_length - len(vals[i]))
                    ret[key] = torch.tensor(ret[key], dtype=torch.long)

        # pre calc the rel mat to reduce the training time
        rel_pos_mat = pre_calc_rel_mat(
            position_ids = ret['position_ids'],
            segment_ids = ret['segment_ids']
        )
        ret['rel_pos_mat'] = rel_pos_mat
        del ret['segment_ids']

        return ret