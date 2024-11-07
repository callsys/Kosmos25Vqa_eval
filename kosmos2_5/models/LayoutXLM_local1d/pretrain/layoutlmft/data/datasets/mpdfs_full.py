import mmap
import math
import json
import os
import random
import requests
from transformers.utils import logging
from tqdm import tqdm
import threading

from torch.utils.data import Dataset

logger = logging.get_logger(__name__)

class mpdfsFullDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.data_dir = args.data_dir
        self.index_path = os.path.join(self.data_dir, "index.txt")

        self.rng = random.Random()
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.max_len_single_sentence

        assert 256 <= self.max_length <= 512

        self.threadlocal = threading.local()

        self.offsets = self._build_index(self.index_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            f = open(self.index_path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.threadlocal.handles = [f, mm]
            if (
                self.index_path.endswith(".gz")
                or self.index_path.endswith(".bz")
                or self.index_path.endswith(".bz2")
            ):
                raise NotImplementedError(
                    "Compressed files are not supported because .seek() would require "
                    "rereading the entire file, making performance too slow."
                )
        return self.threadlocal.handles[-1]

    def _build_index(self, path: str):
        """Build index of start positions of each line."""
        bar = tqdm()
        logger.info(f"Building index for file: {path}")
        f = self._get_mmap()
        f.seek(0)
        offsets = []
        cur = 0
        line_num = 0
        while True:
            bar.update(1)
            line = f.readline()
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
            line_num += 1
        return offsets

    def __len__(self):
        return len(self.offsets)

    def bbox_norm(self, bbox, width, height):
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        x0, y0, x1, y1 = bbox
        x0 = clip(0, int((x0 / width) * 1000), 1000)
        y0 = clip(0, int((y0 / height) * 1000), 1000)
        x1 = clip(0, int((x1 / width) * 1000), 1000)
        y1 = clip(0, int((y1 / height) * 1000), 1000)

        return [x0, y0, x1, y1]

    def random_sample(self, lines, bboxs):
        text_list, bbox_list, pos_list, segment_ids_list = [], [], [], []
        for i in range(len(lines)):
            cur_line, cur_bbox = lines[i], bboxs[i]
            try:
                tokenized_line = self.tokenizer(cur_line, add_special_tokens=False, return_attention_mask=False)
            except:
                logger.info("tokenizer:" + ' '.join(cur_line))
            line_bbox = cur_bbox

            tokenized_line['bbox'] = [line_bbox] * len(tokenized_line['input_ids'])
            tokenized_line['1d_pos'] = [2 + j for j in range(len(tokenized_line['input_ids']))]
            tokenized_line['seg_id'] = [1 + i for j in range(len(tokenized_line['input_ids']))]

            text_list.extend(tokenized_line['input_ids'])
            bbox_list.extend(tokenized_line['bbox'])
            pos_list.extend(tokenized_line['1d_pos'])
            segment_ids_list.extend(tokenized_line['seg_id'])

        assert len(text_list) > 0

        if len(text_list) <= self.max_length:
            start = 0
            end = len(text_list)
        else:
            start = self.rng.randint(0,
                                     len(text_list) - self.max_length - 1)
            end = start + self.max_length

        res_text_list = text_list[start:end]
        res_bbox_list = bbox_list[start:end]
        res_pos_list = pos_list[start:end]
        res_segment_ids_list = segment_ids_list[start:end]
        assert len(res_text_list) == len(res_bbox_list) == len(res_pos_list) == len(res_segment_ids_list)
        assert len(res_text_list) > 0, res_text_list
        assert len(res_text_list) > 100

        return (res_text_list, res_bbox_list, res_pos_list, res_segment_ids_list)

    def normalText(self, t):
        if type(t) is float:
            if t == int(t):
                t = int(t)
        t = str(t)
        return t.strip()

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def rotate(self, data):
        def trans(bbox):
            blocks_width, blocks_height = data['blocks']['width'], data['blocks']['height']

            x0, y0, x1, y1 = bbox

            x, y = [], []
            y.append(x0)
            x.append(blocks_height - y0)
            y.append(x1)
            x.append(blocks_height - y1)

            new_x0, new_x1 = min(x), max(x)
            new_y0, new_y1 = min(y), max(y)
            return [new_x0, new_y0, new_x1, new_y1]

        for i in range(len(data['blocks']['blocks'])):
            for j in range(len(data['blocks']['blocks'][i]['lines'])):
                for k in range(len(data['blocks']['blocks'][i]['lines'][j]['spans'])):
                    data['blocks']['blocks'][i]['lines'][j]['spans'][k]['bbox'] = trans(data['blocks']['blocks'][i]['lines'][j]['spans'][k]['bbox'])
        data['blocks']['width'], data['blocks']['height'] = data['blocks']['height'], data['blocks']['width']
        return data


    def download(self, url):
        qheaders = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64)", "Connection": "close"}
        r = requests.get(url, headers=qheaders)
        return json.loads(str(r.content, 'utf-8'))

    def get_ocr(self, path):
        def get_line_bbox(bbox):
            x0 = min([bbox[i][0] for i in range(len(bbox))])
            y0 = min([bbox[i][1] for i in range(len(bbox))])
            x1 = max([bbox[i][2] for i in range(len(bbox))])
            y1 = max([bbox[i][3] for i in range(len(bbox))])
            return [x0, y0, x1, y1]

        with open(path, 'r', encoding='utf-8') as fr:
            data = json.load(fr)

        if (math.fabs(data['width'] - data['blocks']['width']) <= 2 and math.fabs(
            data['height'] - data['blocks']['height']) <= 2) == False:
            if (math.fabs(data['width'] - data['blocks']['height']) <= 2) and (
                    data['height'] - data['blocks']['width'] <= 2):
                data = self.rotate(data)
        assert math.fabs(data['width'] - data['blocks']['width']) <= 2 and math.fabs(
            data['height'] - data['blocks']['height']) <= 2, "rank:{}:::{}:{}-{}:{}".format(self.args.local_rank, data['width'], data['blocks']['width'], data['height'], data['blocks']['height'])

        width = data['blocks']['width']
        height = data['blocks']['height']

        blocks_lines, blocks_bboxs = [], []
        for block in data['blocks']['blocks']:
            cur_block_lines, cur_block_bboxs = [], []
            for line in block['lines']:
                cur_line_word, cur_line_bbox = [], []
                for span in line['spans']:
                    text = self.normalText(span['text']).encode('utf-8', 'ignore').decode("utf-8").strip()
                    if len(text) == 0: continue
                    cur_line_word.append(text)
                    cur_line_bbox.append(self.bbox_norm(bbox=span['bbox'], width=width, height=height))
                assert len(cur_line_word) == len(cur_line_bbox)
                if len(cur_line_word) == 0: continue
                cur_block_lines.append(' '.join(cur_line_word))
                cur_block_bboxs.append(get_line_bbox(cur_line_bbox))
            assert len(cur_block_lines) == len(cur_block_bboxs)
            blocks_lines.append(cur_block_lines)
            blocks_bboxs.append(cur_block_bboxs)
        assert len(blocks_lines) == len(blocks_bboxs)

        lines, bboxs = [], []
        for i in range(len(blocks_lines)):
            assert len(blocks_lines[i]) == len(blocks_bboxs[i])
            for j in range(len(blocks_lines[i])):
                cur_line = blocks_lines[i][j]
                cur_bbox = blocks_bboxs[i][j]

                # assert len(cur_line) == len(cur_bbox)
                if len(cur_line) == 0: continue

                lines.append(cur_line)
                bboxs.append(cur_bbox)
        assert len(lines) == len(bboxs)
        assert len(lines) > 0

        return lines, bboxs

    def repos_line(self, pos):
        lines = []

        cur_line = [pos[0]]
        for i in range(1, len(pos)):
            if pos[i] != pos[i - 1] + 1:
                lines.append(cur_line)
                cur_line = [pos[i]]
            else:
                cur_line.append(pos[i])
        if len(cur_line) != 0:
            lines.append(cur_line)

        for i in range(len(lines)):
            sub_num = lines[i][0] - 2
            assert sub_num >= 0
            for j in range(len(lines[i])):
                lines[i][j] -= sub_num
            assert lines[i][0] >= 2

        res = []
        for line in lines:
            res += line
        assert len(res) == len(pos)
        return res

    def __getitem__(self, idx):
        flag = False

        while flag == False:
            try:
                f = self._get_mmap()
                f.seek(self.offsets[idx])
                item = f.readline().decode("utf-8").strip().split("+")

                path = os.path.join(
                    self.data_dir,
                    "data",
                    item[0],
                    "{}.json".format(item[1]))

                lines, bboxs = self.get_ocr(path)
                text_list, bbox_list, pos, seg = self.random_sample(lines, bboxs)
                flag = True
            except Exception as e:
                # logger.warning(e)
                idx = self.rng.randint(0, len(self.offsets) - 1)

        model_input = self.tokenizer.prepare_for_model(
            text_list,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        sub_num = seg[0] - 3
        for i in range(len(seg)): seg[i] -= sub_num

        # sub_num = seg[0]
        # for i in range(len(seg)): seg[i] = seg[i] - sub_num + 2
        pos = self.repos_line(pos)

        bbox_list = [[0] * 4] + bbox_list + [[1000] * 4]
        pos = [2] + pos + [2]
        seg = [2] + seg + [seg[-1] + 1]

        input_ids = model_input['input_ids']
        assert len(input_ids) == len(bbox_list)
        doc = {
            "input_ids": input_ids,
            "bbox": bbox_list,
            "position_ids": pos,
            "segment_ids": seg,
        }

        return doc
