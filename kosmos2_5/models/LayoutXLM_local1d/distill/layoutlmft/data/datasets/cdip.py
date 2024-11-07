import json
import os
import random

from torch.utils.data import Dataset

from ...utils import FileSystem


class CdipDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.data_dir = args.data_dir
        self.segment_layout_embedding = args.segment_layout_embedding
        self.fs = FileSystem()
        self.file_index = list(range(args.dataset_size if args.dataset_size else 42948004))
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.max_len_single_sentence
        assert 256 <= self.max_length <= 512

    def __len__(self):
        return len(self.file_index)

    def _clip(self, min_num, num, max_num):
        return max(min_num, min(num, max_num))

    def load_doc(self, fpath):
        with self.fs.open(fpath) as f:
            page = json.load(f)
        input_ids, bbox = [], []
        for line in page["lines"]:
            if self.segment_layout_embedding:
                input_ids = self.tokenizer(line["text"], truncation=False, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                x0 = min(line["boundingBox"][0::2])
                y0 = min(line["boundingBox"][1::2])
                x1 = max(line["boundingBox"][::2])
                y1 = max(line["boundingBox"][1::2])
                assert x1 >= x0 and y1 >= y0
                bbox = [[x0, y0, x1, y1]] * len(input_ids)
            else:
                for word in line["words"]:
                    x = [word["boundingBox"][i] for i in range(0, len(word["boundingBox"]), 2)]
                    y = [word["boundingBox"][i] for i in range(1, len(word["boundingBox"]), 2)]
                    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
                    assert x1 >= x0 and y1 >= y0
                    cur_word = word["text"]
                    sub_token_ids = self.tokenizer(
                        cur_word, truncation=False, add_special_tokens=False, return_attention_mask=False
                    )["input_ids"]
                    input_ids.extend(sub_token_ids)
                    bbox.extend([[x0, y0, x1, y1]] * len(sub_token_ids))
        return {"input_ids": input_ids, "bbox": bbox, "width": page["width"], "height": page["height"]}

    def random_sample(self, doc):
        length = len(doc["input_ids"])
        if length <= self.max_length:
            return doc
        start_index = random.randint(0, length - self.max_length)
        doc["input_ids"] = doc["input_ids"][start_index : start_index + self.max_length]
        doc["bbox"] = doc["bbox"][start_index : start_index + self.max_length]
        return doc

    def normalize_bbox(self, doc):
        doc["bbox"] = [
            [
                self._clip(0, int(1000 * b[0] // doc["width"]), 1000),
                self._clip(0, int(1000 * b[1] // doc["height"]), 1000),
                self._clip(0, int(1000 * b[2] // doc["width"]), 1000),
                self._clip(0, int(1000 * b[3] // doc["height"]), 1000),
            ]
            for b in doc["bbox"]
        ]
        del doc["width"]
        del doc["height"]
        return doc

    def build_inputs(self, doc):
        doc["input_ids"] = [self.tokenizer.cls_token_id] + doc["input_ids"] + [self.tokenizer.sep_token_id]
        doc["bbox"] = [[0, 0, 0, 0]] + doc["bbox"] + [[0, 0, 0, 0]]
        return doc

    def map_fpath(self, file_index):
        file_path = f"coco/annotations/{file_index:012d}.json"
        return os.path.join(self.data_dir, file_path)

    def __getitem__(self, index):
        fpath = self.map_fpath(self.file_index[index])
        while self.fs.exists(fpath) is False:
            replace_idx = random.randint(0, self.__len__() - 2)
            if replace_idx >= index:
                replace_idx += 1
            fpath = self.map_fpath(self.file_index[replace_idx])
        doc = self.load_doc(fpath)
        doc = self.random_sample(doc)
        doc = self.normalize_bbox(doc)
        doc = self.build_inputs(doc)
        return doc
