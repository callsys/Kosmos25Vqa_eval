'''
	inference code from: kosmos2.5
	anls metric code from: https://github.com/allanj/LayoutLMv3-DocVQA/blob/master/src/utils.py
'''
import json
import glob
import argparse
import re
import os
import ast
import tqdm
import base64
import imageio
import io
import numpy as np
import tiktoken
import torch
import string
import textdistance as td

from omegaconf import OmegaConf
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from kosmos2_5 import GenerationTask
from PIL import Image
from transformers import AutoProcessor

def parse_list(arg):
    try:
        parsed_list = ast.literal_eval(arg)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Argument must be a list formatted as '[value1, value2, ...]'")

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt", "-c", type=str, default="")
	parser.add_argument("--use_preprocess", action='store_true', default=False, help="")
	parser.add_argument("--hw_ratio_adj_upper_span", type=parse_list, default=[1.5, 5])
	parser.add_argument("--hw_ratio_adj_lower_span", type=parse_list, default=[0.5, 1.0])

	args = parser.parse_args()
	assert os.path.exists(args.ckpt), "Ckpt does not exist."
	return args

def init(args):
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
	cfg['common_eval']['path'] = args.ckpt
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
			model.cuda()
		model.prepare_for_inference_(cfg)

	generator = task.build_generator(models, cfg.generation)
	generator.max_len_a = 1.0
	tokenizer = tiktoken.get_encoding("cl100k_base")
	image_processor = AutoProcessor.from_pretrained("google/pix2struct-large", is_vqa=False)

	return task, models, generator, image_processor, dictionary, tokenizer

def build_data(args, image, question, image_processor, tokenizer, dictionary):
	bos_id = dictionary.bos()
	eos_id = dictionary.eos()
	boi_id = dictionary.index("<image>")
	eoi_id = dictionary.index("</image>")
	image_feature_length = 2048


	token, img_gpt_input_mask, segment_token = [], [], []

	token.append(bos_id)
	img_gpt_input_mask.append(0)
	segment_token.append(0)

	# image = Image.open(image_path).convert("RGB")

	raw_width, raw_height = image.width, image.height
	if args.use_preprocess:
		ratio = raw_height / raw_width

		if args.hw_ratio_adj_upper_span[1] > ratio > args.hw_ratio_adj_upper_span[0]:
			new_width = int(raw_height / args.hw_ratio_adj_upper_span[0])
			image = image.resize((new_width, raw_height))
		elif args.hw_ratio_adj_lower_span[1] > ratio > args.hw_ratio_adj_lower_span[0]:
			new_height = (int(raw_width * args.hw_ratio_adj_lower_span[1]))
			image = image.resize((raw_width, new_height))

	img_res = image_processor(images=image, return_tensors="pt", max_patches=4096)
	width = img_res['width'][0]
	height = img_res['height'][0]
	img_src_token = img_res['flattened_patches'][0]
	img_attn_mask = img_res['attention_mask'][0]
	token.extend([boi_id] + list(range(4, image_feature_length + 4)) + [eoi_id])
	img_gpt_input_mask.extend([0] + [1] * image_feature_length + [0])
	segment_token.extend([1] + [1] * image_feature_length + [1])

	def text_transform(lines):
		fs_dict = dictionary
		ids = [dictionary.index("<md>")]
		ids += [fs_dict.index(str(token)) for token in tokenizer.encode(lines)]
		return ids

	# text_token = [dictionary.index("<md>")]
	question = "You're a smart document AI assistant. Based on the given image, please answer the question given by human.\n" \
			   "Human :" + question + "\n" + "Assistant :"
	text_token = text_transform(question)

	token += text_token
	img_gpt_input_mask += [0] * len(text_token)
	segment_token += [0] * len(text_token)
	# token.append(eos_id)
	assert len(token) == len(img_gpt_input_mask) == len(segment_token)

	token = torch.LongTensor(token)
	img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)
	segment_token = torch.LongTensor(segment_token)
	lengths = torch.LongTensor([t.numel() for t in token])

	return token.unsqueeze(0), lengths, img_src_token.unsqueeze(0), img_attn_mask.unsqueeze(0), img_gpt_input_mask.unsqueeze(0), segment_token.unsqueeze(0), width, height, raw_width, raw_height

def get_vqa_res(tokenizer, tokens, raw_width, raw_height):
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

		md = md.split("Assistant :")[-1]

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
	tokens = tokens[tokens.index('</image>') + 2:tokens.index('</s>')]
	md = tokenizer.decode([int(t) for t in tokens])
	md = md_post_process(md)
	json_data = get_json_format(md, raw_width, raw_height)
	return json_data

def anls_metric_str(predictions, gold_labels, tau=0.5, rank=0):
    res = []
    """
    predictions: List[List[int]]
    gold_labels: List[List[List[int]]]: each instances probably have multiple gold labels.
    """
    for i, (pred, golds) in enumerate(zip(predictions, gold_labels)):
        max_s = 0
        for gold in golds:
            dis = td.levenshtein.distance(pred.lower(), gold.lower())
            max_len = max(len(pred), len(gold))
            if max_len == 0:
                s = 0
            else:
                nl = dis / max_len
                s = 1-nl if nl < tau else 0
            max_s = max(s, max_s)
        res.append(max_s)
    return res, sum(res)/len(res)

def main():
	args = get_args()
	task, models, generator, image_processor, dictionary, tokenizer = init(args)

	vqa_data = json.load(open("/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/vqa/test.json"))
	data_dir = "/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/"

	samples = vqa_data[0]['source']

	gt_pred_pairs = []

	for idx, sample in enumerate(tqdm.tqdm(samples)):

		file_path = os.path.join(data_dir, sample)
		if not os.path.exists(file_path):
			print('| file {} not exists'.format(file_path), flush=True)

		with open(file_path, 'r', encoding='utf8') as f:
			lines = f.read().strip().split('\n')
		line = lines[0]

		assert len(line.strip()) != 0
		doc_str = json.loads(line.strip())

		assert "question" in doc_str.keys()
		question = doc_str["question"]
		answers = doc_str["answers"]

		image = imageio.imread(io.BytesIO(base64.b64decode(doc_str['image'])), pilmode='RGB')

		pil_img = Image.fromarray(image)

		src_tokens, src_lengths, img_src_token, img_attn_mask, img_gpt_input_mask, segment_token, p2s_resized_width, p2s_resized_height, raw_width, raw_height = build_data(
			args, pil_img, question, image_processor, tokenizer, dictionary)

		src_tokens = src_tokens.cuda()
		src_lengths = src_lengths.cuda().half()
		img_src_token = img_src_token.cuda().half()
		img_attn_mask = img_attn_mask.cuda().half()
		img_gpt_input_mask = img_gpt_input_mask.cuda().half()
		segment_token = segment_token.cuda()

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

		translations = task.inference_step(
			generator, models, sample, constraints=None
		)

		tokens = []
		for tid in translations[0][0]["tokens"].int().cpu().tolist():
			cur_id = dictionary[tid]
			tokens.append(cur_id)

		result = get_vqa_res(tokenizer, tokens, raw_width, raw_height)

		pred_answer = result["results"]

		gt_pred_pairs.append([answers, pred_answer])


		if (idx+1) % 20 == 0:
			preds = [el[1] for el in gt_pred_pairs]
			gts = [el[0] for el in gt_pred_pairs]
			_, anls = anls_metric_str(preds, gts)
			print(f"\nANLS score at evaluation step ({idx+1}/{len(samples)}) : {anls}\n")






if __name__ == '__main__':
	main()
