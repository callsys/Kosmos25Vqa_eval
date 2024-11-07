import json
import jsonlines
from tqdm import tqdm
import os
from icecream import ic
from evaluation.benchmarks_eval import (llm_text_localization_eval, llm_textcaps_textvqa_eval, llm_benchmark_eval)
from evaluation.due_benchmarks_eval import llm_duebenchmark_eval
import argparse

import json
import glob
import argparse
import re
import os
import ast
import base64
import imageio
import io
import numpy as np
import tiktoken
import torch
import cv2
import multiprocessing
import concurrent.futures as futures

from itertools import groupby
from omegaconf import OmegaConf
from functools import partial
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from kosmos2_5 import GenerationTask
from PIL import Image
from transformers import AutoProcessor
from kosmos2_5.data.templates.kosmos_template import KosmosTemplate


def read_jsonl(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            lines.append(line)
    return lines


def save_jsonl(data, filename, print_log=True):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))

    if print_log:
        print('save %d samples to %s' % (len(data), filename))


class Kosmos25Inference:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device
        task, models, generator, image_processor, dictionary, tokenizer = self.init(args)

        self.task = task
        self.models = models
        self.generator = generator
        self.image_processor = image_processor
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.use_preprocess = False
        self.hw_ratio_adj_upper_span = [1.5, 5]
        self.hw_ratio_adj_lower_span = [0.5, 1.0]
        self.template = KosmosTemplate(tokenizer=self.tokenizer,
                                       dictionary=self.dictionary)

    def init(self, args):
        cfg = {
            '_name': None,
            'common': {
                'fp16': True,
            },
            'common_eval': {
                '_name': None,
                'path': None,
                'post_process': 'sentencepiece',
                'quiet': False,
                'model_overrides': '{}',
                'results_path': None,
                'is_moe': False
            },
            'generation': {
                '_name': None,
                'beam': 1,
                'nbest': 1,
                'max_len_a': 0.0,
                'max_len_b': 4000,
                'min_len': 1,
                'match_source_len': False,
                'unnormalized': False,
                'no_early_stop': False,
                'no_beamable_mm': False,
                'lenpen': 1.0,
                'unkpen': 0.0,
                'replace_unk': None,
                'sacrebleu': False,
                'score_reference': False,
                'prefix_size': 0,
                'no_repeat_ngram_size': 0,
                'sampling': False,
                'sampling_topk': -1,
                'sampling_topp': -1.0,
                'constraints': None,
                'temperature': 1.0,
                'diverse_beam_groups': -1,
                'diverse_beam_strength': 0.5,
                'diversity_rate': -1.0,
                'print_alignment': None,
                'print_step': False,
                'lm_path': None,
                'lm_weight': 0.0,
                'iter_decode_eos_penalty': 0.0,
                'iter_decode_max_iter': 10,
                'iter_decode_force_max_iter': False,
                'iter_decode_with_beam': 1,
                'iter_decode_with_external_reranker': False,
                'retain_iter_history': False,
                'retain_dropout': False,
                'retain_dropout_modules': None,
                'decoding_format': None,
                'no_seed_provided': False
            },
            'task': {
                '_name': 'generation',
                'data': '',
                'required_batch_size_multiple': 1,
                'dict_path': './dict.txt',
            },
        }
        cfg['common_eval']['path'] = args.model_path
        cfg = OmegaConf.create(cfg)

        utils.import_user_module(cfg.common)
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)

        use_cuda = True

        task = tasks.setup_task(cfg.task)
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)

        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix='',
            strict=True,
            num_shards=1,
        )

        dictionary = task.source_dictionary

        for model in models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if use_cuda:
                model.to(self.device)
            model.prepare_for_inference_(cfg)

        generator = task.build_generator(models, cfg.generation)
        generator.max_len_a = 1.0
        tokenizer = tiktoken.get_encoding("cl100k_base")
        image_processor = AutoProcessor.from_pretrained("google/pix2struct-large", is_vqa=False)

        return task, models, generator, image_processor, dictionary, tokenizer

    def build_data(self, image, question):
        raw_width, raw_height = image.width, image.height
        data_dict = {}
        img_res = self.image_processor(images=image, return_tensors="pt", max_patches=4096)
        data_dict.update(dict(img_res))
        data_dict["has_image"] = 1

        conversations = [{"from": "human", "value": question}, {"from": "gpt", "value": None}]
        template_dict = self.template.encode(conversations, mode="eval", **data_dict)
        data_dict.update(template_dict)

        img_src_token = data_dict['flattened_patches'][0]
        img_attn_mask = data_dict['attention_mask'][0]
        width = data_dict['width'][0]
        height = data_dict['height'][0]

        token = torch.LongTensor(data_dict["text_tokens"].tolist())
        img_gpt_input_mask = torch.LongTensor(data_dict["text_input_mask"].tolist())
        segment_token = torch.LongTensor(data_dict["segment_ids"].tolist())
        assert len(token) == len(img_gpt_input_mask) == len(segment_token)

        lengths = torch.LongTensor([t.numel() for t in token])

        return token.unsqueeze(0), lengths, img_src_token.unsqueeze(0), img_attn_mask.unsqueeze(
            0), img_gpt_input_mask.unsqueeze(0), segment_token.unsqueeze(0), width, height, raw_width, raw_height

    def get_vqa_res(self, tokenizer, tokens, raw_width, raw_height):
        def md_pre_process(tokens):
            return tokens

        def md_post_process(md):
            md = md.replace('<br>', '\n')
            lines = md.split('\n')
            new_lines = []
            for i in range(len(lines)):
                text = lines[i].strip()
                new_lines.append(text)
            md = '\n'.join(new_lines)
            md = re.sub('\n{2,}', '\n\n', md).strip()

            md = md.split("ASSISTANT:")[-1]

            return md

        def get_json_format(md, raw_width, raw_height):
            json_res = {
                'model': "kosmos 2.5",
                'task': "markdown",
                'width': raw_width,
                'height': raw_height,
                "results": md,
            }
            return json_res

        tokens = md_pre_process(tokens)
        tokens = tokens[tokens.index('<md>') + 2:tokens.index('</s>')]
        md = tokenizer.decode([int(t) for t in tokens])
        md = md_post_process(md)
        json_data = get_json_format(md, raw_width, raw_height)
        return json_data

    def __call__(self, image ,query):
        pil_img = Image.fromarray(cv2.imread(image)).convert("RGB")

        raw_width, raw_height = pil_img.width, pil_img.height
        if self.use_preprocess:
            ratio = raw_height / raw_width
            if self.hw_ratio_adj_upper_span[1] > ratio > self.hw_ratio_adj_upper_span[0]:
                new_width = int(raw_height / self.hw_ratio_adj_upper_span[0])
                pil_img = pil_img.resize((new_width, raw_height))
            elif self.hw_ratio_adj_lower_span[1] > ratio > self.hw_ratio_adj_lower_span[0]:
                new_height = (int(raw_width * self.hw_ratio_adj_lower_span[1]))
                pil_img = pil_img.resize((raw_width, new_height))

        image_processor = self.image_processor
        tokenizer = self.tokenizer
        dictionary = self.dictionary

        src_tokens, src_lengths, img_src_token, img_attn_mask, img_gpt_input_mask, segment_token, p2s_resized_width, p2s_resized_height, raw_width, raw_height = self.build_data(
            pil_img, query)

        src_tokens = src_tokens.to(self.device)
        src_lengths = src_lengths.to(self.device).half()
        img_src_token = img_src_token.to(self.device).half()
        img_attn_mask = img_attn_mask.to(self.device).half()
        img_gpt_input_mask = img_gpt_input_mask.to(self.device).half()
        segment_token = segment_token.to(self.device)

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "image": img_src_token,
                'image_attention_masks': img_attn_mask,
                "segment_tokens": segment_token,
                "img_gpt_input_mask": img_gpt_input_mask,
            },
        }

        translations = self.task.inference_step(
            self.generator, self.models, sample, constraints=None
        )

        tokens = []
        for tid in translations[0][0]["tokens"].int().cpu().tolist():
            cur_id = dictionary[tid]
            tokens.append(cur_id)

        result = self.get_vqa_res(tokenizer, tokens, raw_width, raw_height)

        pred_answer = result["results"].strip()
        return pred_answer


class ParallelKosmos25Inference:
    def __init__(self, args):
        self.args = args
        self.downstream_dir = args.downstream_dir

        multiprocessing.set_start_method('spawn', force=True)

        self.gpu_ids = list(range(torch.cuda.device_count()))

    def process_wrapper(self, args):
        total_processes = args.get("max_workers", 1)
        idx = args.get("pid", 0)
        samples = args.get("samples", [])
        task_func = args.get("task_func", None)
        gpu_id = self.gpu_ids[idx % len(self.gpu_ids)]
        if "model" in args:
            model = args.get("model")
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            gpu_model = model(device=device)
            task_func = partial(task_func, model=gpu_model)

        bsamples = [[sample[0] for sample in bsample[1]] for bsample in groupby(zip(samples, range(len(samples))), key = lambda x:x[1]//self.batch_size)]
        infer_results = []
        for bsample in tqdm(bsamples, desc=f"Process {idx + 1}/{total_processes}"):
            try:
                results = task_func(bsample=bsample)
                infer_results.extend(results)
            except:
                print(f"ERROR sample on cuda:{gpu_id}")
                raise
        return infer_results

    def run_single_process(self, model=None, bsample=None):
        assert bsample is not None
        infer_results = []
        for sample in bsample:
            image = os.path.join(self.downstream_dir, sample['image'][0])
            assert os.path.exists(image)
            question = sample['messages'][0]
            answer = sample['messages'][1]
            assert question['role'] == 'user'
            assert answer['role'] == 'assistant'
            query = question['content'].replace('<|image|>', '')
            gt_answer = answer['content']
            model_answer = model(image, query)
            sample['model_answer'] = model_answer
            sample['gt_answer'] = gt_answer
            ic(query, model_answer, gt_answer)
            infer_results.append(sample)
        return infer_results

    def __call__(self, test_samples):
        self.batch_size = 1
        max_workers = len(self.gpu_ids)

        for i, sample in enumerate(test_samples):
            sample["id"] = i

        process_args = [{"model": partial(Kosmos25Inference, args=self.args),
                         "samples": test_samples[i::max_workers],
                         "pid": i,
                         "max_workers": max_workers,
                         "task_func": self.run_single_process} for i in range(max_workers)]

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_wrapper, process_args))

        infer_results = []
        for result in results:
            infer_results.extend(result)

        infer_results = sorted(infer_results, key=lambda x:x["id"])

        return infer_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='docowl1.5 benchmark evaluation')
    parser.add_argument('--model_path', type=str, help='the directory path of model')
    parser.add_argument('--pred_path', type=str, default="", help='the directory path of model')
    parser.add_argument('--dataset', type=str,
                        choices=['DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 'DeepForm', 'KleisterCharity',
                                 'TabFact',
                                 'ChartQA', 'TextVQA', 'TextCaps', 'VisualMRC'])
    parser.add_argument('--downstream_dir', type=str, help='the directory path of DocDownstream-1.0')
    parser.add_argument('--save_dir', type=str, help='the directory to save predictions of the model')
    args = parser.parse_args()

    model_path = args.model_path
    dataset = args.dataset
    downstream_dir = args.downstream_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(args.pred_path) != 0:
        pred_path = args.pred_path
    else:
        test_path = os.path.join(downstream_dir, 'test', dataset + '_test.jsonl')
        save_path = os.path.join(save_dir, dataset + '_test_pred.jsonl')

        if os.path.exists(save_path):
            print(save_path + ' exists, skip inference. ')
        else:
            evaluation_task = ParallelKosmos25Inference(args)
            print('load model from ', model_path)
            test_samples = read_jsonl(test_path)
            infer_results = evaluation_task(test_samples)
            save_jsonl(infer_results, save_path)

        # calculate metrics
        pred_path = save_path

    if not os.path.exists(pred_path):
        print('not exists:', pred_path)
        exit(0)

    meta_dir = os.path.join(downstream_dir, 'meta')

    if dataset in ['DeepForm', 'DocVQA', 'InfographicsVQA', 'KleisterCharity', 'WikiTableQuestions']:
        llm_duebenchmark_eval(dataset_name=dataset, split='test', llm_pred_path=pred_path, meta_dir=meta_dir)
    elif dataset in ['TabFact']:
        llm_benchmark_eval(metric_names=['ExactAccuracy'], result_path=pred_path, save_each_eval=True)
    elif dataset in ['ChartQA']:
        llm_benchmark_eval(metric_names=['RelaxedAccuracy'], result_path=pred_path, save_each_eval=True)
    elif dataset in ['TextCaps', 'TextVQA']:
        llm_textcaps_textvqa_eval(result_path=pred_path, dataset=dataset, split='test', meta_dir=meta_dir)
    elif dataset in ['VisualMRC']:
        llm_benchmark_eval(metric_names=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'Meteor', 'RougeL', 'CIDEr'],
                           result_path=pred_path, save_each_eval=True)

    print('==============================================')






