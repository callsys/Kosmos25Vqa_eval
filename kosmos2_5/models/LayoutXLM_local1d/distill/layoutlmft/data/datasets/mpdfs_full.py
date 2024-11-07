import torch
from typing import Union
import gc
import math
import json
import os
import random
from transformers.utils import logging
import linecache
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
import requests
import traceback
from multiprocessing import Manager

import mmap
import threading

logger = logging.get_logger(__name__)
from transformers.file_utils import TRANSFORMERS_CACHE
DATA_ROOT_PAYH = "/mnt/localdata2/users/tengchaolv/datasets/multilingualPDF"

def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


class mpdfsFullDataset(Dataset):
    # def load_index(self, path):
    #     data = []
    #     index = 0
    #     with open(path, 'r') as f:
    #         for line in f:
    #             index += 1
    #             if index >= 10000: continue
    #             url, pno, la = line.strip().split('+')
    #             data.append([url, pno])
    #     return data
    def _get_mmap(self, path):
        if not hasattr(self.threadlocal, "handles"):
            f = open(path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.threadlocal.handles = [f, mm]
            if (
                path.endswith(".gz")
                or path.endswith(".bz")
                or path.endswith(".bz2")
            ):
                raise NotImplementedError(
                    "Compressed files are not supported because .seek() would require "
                    "rereading the entire file, making performance too slow."
                )
        return self.threadlocal.handles[-1]

    def _build_index(self, path: str):
        """Build index of start positions of each line."""
        logger.info(f"Building index for file: {path}")
        f = self._get_mmap(path)
        f.seek(0)
        offsets = []
        cur = 0

        bar = tqdm()
        while True:
#            if len(offsets) > 10000:
#                break
            line = f.readline()
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
            bar.update()
        return offsets

    def guess_sample(self):
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + 1)
        indices = torch.randperm(self.length, generator=generator).tolist()

        
        gpu_num=64
        batch_size_per_gpu=4
        acc_steps=2

#        self.rank_index_dict = {index: 0 for index in indices[self.args.local_rank:self.length:gpu_num][:self.args.skip_sample_num * batch_size_per_gpu * acc_steps]}
        self.rank_index_dict = {index: 0 for index in indices[self.args.skip_sample_num * batch_size_per_gpu * acc_steps * gpu_num:]}

        self.skip_flag = True

    def __init__(self, args, tokenizer):
        self.args = args

        self.data_dir = args.data_dir
        self.no_img = True
        self.rng = random.Random()

        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.max_len_single_sentence
        assert 256 <= self.max_length <= 512

        self.threadlocal = threading.local()

        self.index_file = os.path.join(args.data_dir, "index_block_la_short.txt")
#         index_file = os.path.join(args.data_dir, "index_block_la_short_76las.txt")
#        index_file = os.path.join(args.data_dir, "index_block_la_short_8las.txt")
#         index_file = "/mnt/localdata3/users/tengchaolv/prj/layoutxlm/dataset/tmp.txt"
#        self.index_file = "/mnt/localdata3/users/tengchaolv/tmp/index_block_la_short.txt"
        # self.examples = linecache.getlines(index_file)
        # self.length = len(self.examples)
        # examples = linecache.getlines(index_file)
        # self.examples = np.array(examples)
        # del examples
        # gc.collect()
        # self.length = len(self.examples)
        self.offsets = np.array(self._build_index(self.index_file))
        self.length = len(self.offsets)
        # print('w')

#        for i in tqdm(range(len(examples)), desc='trans to seq'):
#            examples[i] = string_to_sequence(examples[i].strip(), dtype=np.int8)
#
#        self.example_v, self.example_o = pack_sequences(examples)
#        del examples
        gc.collect()

#        self.guess_sample()


    def __len__(self):
        return self.length

    def bbox_norm(self, bbox, width, height):
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        x0, y0, x1, y1 = bbox
        x0 = clip(0, int((x0 / width) * 1000), 1000)
        y0 = clip(0, int((y0 / height) * 1000), 1000)
        x1 = clip(0, int((x1 / width) * 1000), 1000)
        y1 = clip(0, int((y1 / height) * 1000), 1000)

        return [x0, y0, x1, y1]

    # def random_sample(self, page, img):
    def random_sample(self, lines, bboxs):
        text_list, bbox_list, pos_list, segment_ids_list = [], [], [], []
        for i in range(len(lines)):
            cur_line, cur_bbox = lines[i], bboxs[i]
            assert len(cur_line) == len(cur_bbox)
            try:
                tokenized_line = self.tokenizer(' '.join(cur_line), add_special_tokens=False, return_attention_mask=False)
            except:
                logger.info("tokenizer:" + ' '.join(cur_line))
                assert 1 == 2, ' '.join(cur_line)
            line_bbox = self.get_line_bbox(cur_bbox)[0]

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

        assert len(res_text_list) > 100, "skip sample because too short: {}".format(len(res_text_list))

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

    def get_actual_path(self, example):
        la, raw_prefix = example[0].split('/')
        prefix = raw_prefix.split('-')[0]
        mid_path = '/'.join([c for c in prefix])
        path = os.path.join(DATA_ROOT_PAYH, la, mid_path, raw_prefix, "{}.json".format(example[1]))
        return path

    def get_ocr(self, example):
        # tmp_fields = example[0].split('/')
        # assert len(tmp_fields) == 5, "fields != 5"
        url, pno, la = example.strip().split('+')
        example = [url, pno]
#        path = self.get_actual_path(example)
        path = os.path.join(self.data_dir, "pdf_parser_blocks", example[0], "{}.json".format(example[1]))
        # data = self.download(url)

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
                    text = self.normalText(span['text']).encode('utf-8','ignore').decode("utf-8").strip()
                    if len(text) == 0: continue
                    cur_line_word.append(text)
                    cur_line_bbox.append(self.bbox_norm(bbox=span['bbox'], width=width, height=height))

                assert len(cur_line_word) == len(cur_line_bbox)
                if len(cur_line_word) == 0: continue
                cur_block_lines.append(cur_line_word)
                cur_block_bboxs.append(cur_line_bbox)
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

                assert len(cur_line) == len(cur_bbox)
                if len(cur_line) == 0: continue

                lines.append(cur_line)
                bboxs.append(cur_bbox)
        assert len(lines) == len(bboxs)
        assert len(lines) > 0

        if la in ["ja", "zh", "th", "sa"]:
            total_length = sum([len("".join(lines[i]))for i in range(len(lines))])
        else:
            total_length = sum([len(" ".join(lines[i]).split())for i in range(len(lines))])
        line_avg_length = total_length / len(lines)
#        assert total_length > 30, "skip sample {} because too short: {}".format(path, total_length)
#        assert line_avg_length > 1.5, "skip sample {} because small avg length: {}".format(path, line_avg_length)

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

    def __getitem__(self, index):
        flag = False

        while flag == False:
            try:
#                example = unpack_sequence(self.example_v, self.example_o, index)
#                example = sequence_to_string(example)
                f = self._get_mmap(self.index_file)
                f.seek(self.offsets[index])
                item = f.readline().decode("utf-8")
                lines, bboxs = self.get_ocr(item)


                # lines, bboxs = self.get_ocr(self.examples[index])
                text_list, bbox_list, pos, seg = self.random_sample(lines, bboxs)

#                del example
                flag = True

            except Exception as e:
                flag = False
#                logger.warning(e)
#               logger.warning("rank:{}".format(self.args.local_rank))
#                traceback.print_exc()
#                try:
#                    del example
#                except:
#                    continue

                index = self.rng.randint(0, self.length - 1)

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

        if self.tokenizer.num_special_tokens_to_add() == 2:
            bbox_list = [[0] * 4] + bbox_list + [[1000] * 4]
            pos = [2] + pos + [2]
            seg = [2] + seg + [seg[-1] + 1]
        else:
            raise NotImplementedError('num_special_tokens_to_add is not 2.')

        input_ids = model_input['input_ids']
        assert len(input_ids) == len(bbox_list)
        doc = {
            "input_ids": input_ids,
            "bbox": bbox_list,
            "position_ids": pos,
            "segment_ids": seg,
        }

        return doc
