import json
import os
import random
from transformers.utils import logging

from torch.utils.data import Dataset
import deepdish as dd
import traceback
from operator import itemgetter
import torch

logger = logging.get_logger(__name__)
# from ...utils import FileSystem


class mpdfsDataset(Dataset):
    def load_index(self, path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                lang, url, fname, pid = line.split()
                data.append((fname, pid))
        return data

    def __init__(self, args, tokenizer):
        self.data_dir = args.data_dir
        self.no_img = True
        self.rng = random.Random()

        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.max_len_single_sentence
        # self.max_seq_length = 512
        assert 256 <= self.max_length <= 512

        index_file = os.path.join(args.data_dir, "mpdf_datasets", "train.txt")
        self.examples = self.load_index(index_file)

    def __len__(self):
        return len(self.examples)

    def load_oneocr_example(self, fname):
        ocr_path = f'{self.data_dir}/cdip_msocr/cdip-images/{fname}'
        img_path = f"{self.data_dir}/cdip-images/{fname.replace('.ocr.json', '.tif')}"
        try:
            with open(ocr_path, 'r', encoding='utf8') as f:
                ocr = json.load(f)
            img = None
            # if not self.no_img:
            #     img_dir, img_name = os.path.split(img_path)
            #     img_basename = os.path.splitext(img_name)[0]
            #     if not os.path.exists(img_path):
            #         for img_ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
            #             img_path = os.path.join(img_dir,
            #                                     img_basename + '.' + img_ext)
            #             if os.path.exists(img_path):
            #                 img = Image.open(img_path)
            #                 if img.format.lower() in ['jpeg', 'png', 'tiff']:
            #                     break
            #     else:
            #         img = Image.open(img_path)
            #     assert (img.format.lower() in ['jpeg', 'png', 'tiff'])
        except Exception as e:
            logger.warning(e)
            logger.warning('failed to load example: ' + img_path)
            traceback.print_exc()
            ocr, img = None, None
        return ocr, img

    def load_pdf_example(self, fname, pid):
        fid = fname.split('.')[0]
        ocr = dd.io.load(f'{self.data_dir}/mpdfs_preprocessed/{fid}/{pid}.h5')
        img = None
        # if not self.no_img:
        #     img = Image.open(f'{self.data_dir}/mpdfs_png/{fid}/{pid}.png')
        return ocr, img

    def load_example(self, item):
        fname, pid = self.examples[item]
        if fname.endswith('.pdf'):
            ocr, img = self.load_pdf_example(fname, pid)
            return ocr, img, 'pdf'
        elif fname.endswith('.json'):
            ocr, img = self.load_oneocr_example(fname)
            return ocr, img, 'json'
        else:
            logger.warning(f'The format of {fname} is not supported.')
            return None

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def get_bb(self, bb, page_size, normalize=True):
        bbs = [float(j) for j in bb]
        xs, ys = [], []
        for i, b in enumerate(bbs):
            if i % 2 == 0:
                xs.append(b)
            else:
                ys.append(b)
        (width, height) = page_size
        return_bb = [
            self.clamp(min(xs), 0, width - 1),
            self.clamp(min(ys), 0, height - 1),
            self.clamp(max(xs), 0, width - 1),
            self.clamp(max(ys), 0, height - 1),
        ]

        if normalize:
            return_bb = [
                int(1000 * return_bb[0] / width),
                int(1000 * return_bb[1] / height),
                int(1000 * return_bb[2] / width),
                int(1000 * return_bb[3] / height),
            ]
        return return_bb

    def locations_to_bbox(self, bbox_list):
        return (
            min(map(itemgetter(0), bbox_list)),
            min(map(itemgetter(1), bbox_list)),
            max(map(itemgetter(2), bbox_list)),
            max(map(itemgetter(3), bbox_list)),
        )

    def random_sample(self, page, img):
        try:
            assert (len(page['text']) > 0)
        except:
            raise RuntimeError('empty doc-img pair')
        text_list, bbox_list = [], []
        if page['text'] != []:
            height, width = float(page['height']), float(page['width'])
            page_size = (width, height)
            for line_text, line_bbox in zip(page['text'], page['bbox']):
                assert len(line_text) == len(line_bbox)
                tokenized_line = self.tokenizer(line_text, add_special_tokens=False, return_attention_mask=False)

                line_bbox_norm = self.get_bb(
                    self.locations_to_bbox(line_bbox),
                    page_size
                )

                tokenized_line['bbox'] = [line_bbox_norm] * len(tokenized_line['input_ids'])
                text_list.extend(tokenized_line['input_ids'])
                bbox_list.extend(tokenized_line['bbox'])

        if len(text_list) <= self.max_length:
            start = 0
            end = len(text_list)
        else:
            start = self.rng.randint(0,
                                     len(text_list) - self.max_length - 1)
            end = start + self.max_length

        page_img = None
        # if not self.no_img:
        #     page_img = convert_PIL_to_numpy(_apply_exif_orientation(img),
        #                                     format='BGR')
        return (text_list[start:end], bbox_list[start:end], page_img)

    def normalText(self, t):
        if type(t) is float:
            if t == int(t):
                t = int(t)
        t = str(t)
        return t.strip()

    def random_sample_oneocr(self, doc, img):
        available_list = []
        for i in range(len(doc)):
            if not self.no_img:
                try:
                    img.seek(i)
                    assert (len(doc[i]['lines']) > 0)
                except:
                    continue
            available_list.append(i)
        if len(available_list) == 0:
            raise RuntimeError(
                'empty doc-img pair: ' +
                img.filename if hasattr(img, 'filename') else '')
        pid = available_list[self.rng.randint(0, len(available_list) - 1)]
        page = doc[pid]
        lines = page['lines']
        text_list, bbox_list = [], []
        # line_bbox_list = []
        # line_bbox_content_list = []
        # page_token_cnt = 0
        if lines != []:
            height, width = float(page['height']), float(page['width'])
            page_size = (width, height)
            for cnt, line in enumerate(lines):
                line_text, line_bbox = line['text'], line['boundingBox']
                if len(''.join(line_text.split())) == 0: continue
                line_text = line_text.split()
                for i in range(len(line_text)):
                    line_text[i] = self.normalText(line_text[i])
                line_text = ' '.join(line_text)
                tokenized_line = self.tokenizer(line_text, add_special_tokens=False, return_attention_mask=False)
                line_bbox_norm = self.get_bb(
                    line_bbox,
                    page_size
                )

                tokenized_line['bbox'] = [line_bbox_norm] * len(tokenized_line['input_ids'])
                text_list.extend(tokenized_line['input_ids'])
                bbox_list.extend(tokenized_line['bbox'])

        if len(text_list) <= self.max_length:
            start = 0
            end = len(text_list)
        else:
            start = self.rng.randint(0,
                                     len(text_list) - self.max_length - 1)
            end = start + self.max_length

        page_img = None
        # if not self.no_img:
        #     img.seek(pid)
        #     # page_img = Image.fromarray(np.array(img).astype('uint8')).convert('RGB')
        #     page_img = convert_PIL_to_numpy(_apply_exif_orientation(img),
        #                                     format='BGR')

        return (text_list[start:end], bbox_list[start:end], page_img)

    def __getitem__(self, index):
        try:
            ocr, img, format = self.load_example(index)
        except Exception as e:
            # fname, pid = self.examples[item]
            logger.warning(e)
            traceback.print_exc()
            ocr, img, format = None, None, None

        try:
            if ocr is None or not self.no_img and img is None:
                raise ValueError('None format')
            elif format == 'pdf':
                text_list, bbox_list, page_img = self.random_sample(ocr, img)
            elif format == 'json':
                doc = ocr['analyzeResult']['readResults']
                text_list, bbox_list, page_img = self.random_sample_oneocr(doc, img)
            else:
                raise ValueError('None format')
        except Exception as e:
            logger.warning(e)
            traceback.print_exc()
            ocr, img, format = None, None, None

        while ocr is None \
                or not self.no_img and img is None:
            new_item = self.rng.randint(0, len(self.examples) - 1)
            try:
                ocr, img, format = self.load_example(new_item)
                if ocr is None or not self.no_img and img is None:
                    ocr, img, format = None, None, None
                    continue
                if format == 'pdf':
                    text_list, bbox_list, page_img = self.random_sample(
                        ocr, img)
                elif format == 'json':
                    doc = ocr['analyzeResult']['readResults']
                    text_list, bbox_list, page_img = self.random_sample_oneocr(
                        doc, img)
                else:
                    raise ValueError('None format')
            except Exception as e:
                # fname, pid = self.examples[new_item]
                logger.warning(e)
                traceback.print_exc()
                ocr, img, format = None, None, None

        model_input = self.tokenizer.prepare_for_model(
            text_list,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        if self.tokenizer.num_special_tokens_to_add() == 2:
            bbox_list = [[0] * 4] + bbox_list + [[1000] * 4]
        else:
            raise NotImplementedError('num_special_tokens_to_add is not 2.')

        input_ids = model_input['input_ids']
        assert len(input_ids) == len(bbox_list)
        doc = {
            "input_ids": input_ids,
            "bbox": bbox_list,
        }

        return doc