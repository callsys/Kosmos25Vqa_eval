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
	parser.add_argument("--ckpt", "-c", type=str, default="/home/yuzhongzhao/zyz/ckpts/checkpoint_1_500.pt")
	parser.add_argument("--shard", type=str, default="1/4")
	parser.add_argument("--num-sample", type=int, default=-1)
	parser.add_argument("--use_preprocess", action='store_true', default=False, help="")
	parser.add_argument("--hw_ratio_adj_upper_span", type=parse_list, default=[1.5, 5])
	parser.add_argument("--hw_ratio_adj_lower_span", type=parse_list, default=[0.5, 1.0])

	args = parser.parse_args()
	assert os.path.exists(args.ckpt), "Ckpt does not exist."

	print(f"evaluate ({args.ckpt})")

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

	seq_gen_cls = None
	if models[0].__class__.__name__ == "DocGPTmodel":
		from kosmos2_5.tasks.sequence_generator import DocGPTSequenceGenerator
		seq_gen_cls = DocGPTSequenceGenerator

	generator = task.build_generator(models, cfg.generation, seq_gen_cls=seq_gen_cls)
	generator.max_len_a = 1.0
	tokenizer = tiktoken.get_encoding("cl100k_base")
	image_processor = AutoProcessor.from_pretrained("google/pix2struct-large", is_vqa=False)

	from transformers import AutoTokenizer
	model_name_or_path = "/mnt/msranlp/yuzhongzhao/layoutlmv3/"
	layoutlmv3_tokenizer = AutoTokenizer.from_pretrained(
		model_name_or_path,
		add_prefix_space=True,
		use_fast=True)

	return task, models, generator, image_processor, dictionary, tokenizer, layoutlmv3_tokenizer

def build_data(args, image, doc_str, image_processor, tokenizer, layoutlmv3_tokenizer, dictionary):
	def text_transform(doc_str):
		def clip(min_num, num, max_num):
			return min(max(num, min_num), max_num)

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

		layoutlmv3_max_length = 8000
		sep_token_bbox = [1000, 1000, 1000, 1000]
		pad_token_bbox = [0, 0, 0, 0]

		question = doc_str["question"]
		question = "You're a smart document AI assistant. Based on the given image, please answer the question given by human.\n" \
				   "Human :" + question + "\n" + "Assistant :"
		texts = doc_str["texts"]
		bboxs = doc_str["bboxs"]

		# layoutlm : the multimodal tokens =========================================================================
		bboxs_1000 = (np.array(bboxs) * 1000).astype(np.int64).clip(min=0, max=1000).tolist()

		layoutlm_inputs = layoutlmv3_tokenizer(
			[texts],
			padding=False,
			truncation=True,
			max_length=layoutlmv3_max_length,
			return_overflowing_tokens=True,
			is_split_into_words=True,
		)

		word_ids = layoutlm_inputs.word_ids(batch_index=0)
		previous_word_idx = None
		cur_new_bbox_inputs = []
		for word_idx in word_ids:
			if word_idx is None:
				cur_new_bbox_inputs.append(pad_token_bbox)
			elif word_idx != previous_word_idx:
				cur_new_bbox_inputs.append(bboxs_1000[word_idx])
			else:
				cur_new_bbox_inputs.append(bboxs_1000[word_idx])
			previous_word_idx = word_idx
		cur_new_bbox_inputs[-1] = sep_token_bbox
		assert len(cur_new_bbox_inputs) == len(word_ids)
		segment_ids = get_segment_ids(cur_new_bbox_inputs)  # identify the text blocks
		cur_new_position_ids = get_position_ids(segment_ids)  # identify the token position in text blocks

		layoutlm_inputs["segment_ids"] = [segment_ids]
		layoutlm_inputs["bbox"] = [cur_new_bbox_inputs]
		layoutlm_inputs["position_ids"] = [cur_new_position_ids]

		# vqa : the language tokens ================================================================================
		ids = []
		fs_dict = dictionary
		ids += [dictionary.index("<md>")]
		ids += [fs_dict.index(str(token)) for token in tokenizer.encode(question)]

		return ids, layoutlm_inputs

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

	bos_id = dictionary.bos()
	eos_id = dictionary.eos()
	boi_id = dictionary.index("<image>")
	eoi_id = dictionary.index("</image>")
	image_feature_length = 2048

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
	img_src_token = img_res['flattened_patches'][0]
	img_attn_mask = img_res['attention_mask'][0]

	question_tokens, layoutlm_ret = text_transform(doc_str)

	layoutlm_length = len(layoutlm_ret["input_ids"][0])
	question_length = len(question_tokens)

	text_tokens = [bos_id] + [boi_id] * (image_feature_length + 1) + [eoi_id] + \
				  [boi_id] * layoutlm_length + question_tokens
	img_gpt_input_mask = [0] + [0] + [1] * (image_feature_length) + [0] + \
					  [0] * layoutlm_length + [0] * question_length
	layoutlm_gpt_input_mask = [0] + [0] + [0] * (image_feature_length) + [0] + \
						  [1] * layoutlm_length + [0] * question_length
	segment_tokens = [0] + [1] + [1] * (image_feature_length) + [1] + \
				  [0] * layoutlm_length + [0] * question_length  # image tokens


	assert len(text_tokens) == len(img_gpt_input_mask) == len(layoutlm_gpt_input_mask) == len(segment_tokens)

	for key, vals in layoutlm_ret.items():
		layoutlm_ret[key] = torch.tensor(layoutlm_ret[key], dtype=torch.long).cuda()
	rel_pos_mat = pre_calc_rel_mat(
		position_ids=layoutlm_ret['position_ids'],
		segment_ids=layoutlm_ret['segment_ids']
	)
	layoutlm_ret['rel_pos_mat'] = rel_pos_mat.cuda()
	del layoutlm_ret['segment_ids']
	del layoutlm_ret['overflow_to_sample_mapping']

	src_tokens = torch.LongTensor(text_tokens)[None]
	img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)[None]
	layoutlm_gpt_input_mask = torch.LongTensor(layoutlm_gpt_input_mask)[None]
	segment_tokens = torch.LongTensor(segment_tokens)[None]
	# src_lengths = torch.LongTensor([t.numel() for t in text_tokens])[None]
	src_lengths = torch.LongTensor([t.numel() for t in src_tokens[0]])

	src_tokens = src_tokens.cuda()
	src_lengths = src_lengths.cuda().half()
	img_src_token = img_src_token.cuda().half()
	img_attn_mask = img_attn_mask.cuda().half()
	img_gpt_input_mask = img_gpt_input_mask.cuda().half()
	layoutlm_gpt_input_mask = layoutlm_gpt_input_mask.cuda().half()
	segment_tokens = segment_tokens.cuda()

	sample = {
		"net_input": {
			"src_tokens": src_tokens,
			"src_lengths": src_lengths,
			"image": img_src_token[None],
			'image_attention_masks': img_attn_mask[None],
			"segment_tokens": segment_tokens,
			"img_gpt_input_mask": img_gpt_input_mask,
			"layoutlm_gpt_input_mask": layoutlm_gpt_input_mask,
			"layoutlm_input": layoutlm_ret,
		},
	}

	return sample

def get_vqa_res(tokenizer, tokens):
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

	def get_json_format(md):
		json_res = {
			'model': "kosmos 2.5",
			'task': "markdown",
			"results": md,
		}
		return json_res

	tokens = md_pre_process(tokens)
	tokens = tokens[tokens.index('<md>') + 2:tokens.index('</s>')]
	md = tokenizer.decode([int(t) for t in tokens if t.isnumeric()])
	md = md_post_process(md)
	json_data = get_json_format(md)
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
	task, models, generator, image_processor, dictionary, tokenizer, layoutlmv3_tokenizer = init(args)

	vqa_data = json.load(open("/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/vqa/test.json"))
	data_dir = "/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/"

	samples = vqa_data[0]['source']

	shard = args.shard
	cur_shard, num_shard = int(shard.split("/")[0])-1, int(shard.split("/")[1])
	if args.num_sample > 0:
		samples = samples[:args.num_sample]
	# samples = samples[:200]
	samples = samples[cur_shard::num_shard]

	gt_pred_pairs = []
	shard_results = dict()

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
		answers = doc_str["answers"]

		image = imageio.imread(io.BytesIO(base64.b64decode(doc_str['image'])), pilmode='RGB')

		pil_img = Image.fromarray(image)

		sample = build_data(args, pil_img, doc_str, image_processor, tokenizer, layoutlmv3_tokenizer, dictionary)

		translations = task.inference_step(
			generator, models, sample, constraints=None
		)

		tokens = []
		for tid in translations[0][0]["tokens"].int().cpu().tolist():
			cur_id = dictionary[tid]
			tokens.append(cur_id)

		result = get_vqa_res(tokenizer, tokens)

		pred_answer = result["results"].strip()

		gt_pred_pairs.append([answers, pred_answer])
		questionId = doc_str["extra_info"]["questionId"]
		shard_results[questionId] = {"pred_answer": pred_answer, "answers": answers}


		if (idx+1) % 20 == 0:
			preds = [el[1] for el in gt_pred_pairs]
			gts = [el[0] for el in gt_pred_pairs]
			_, anls = anls_metric_str(preds, gts)
			print(f"\nANLS score at evaluation step ({idx+1}/{len(samples)}) : {anls}\n")

	shard_result_dir = os.path.basename(args.ckpt) + f"_{args.num_sample}"
	if not os.path.exists(shard_result_dir):
		os.mkdir(shard_result_dir)
	save_result_path = os.path.join(shard_result_dir, f"evaluate_docvqa_{cur_shard}_{num_shard}.json")
	print(f"dump results to {save_result_path}")
	with open(save_result_path, "w") as fw:
		json.dump(shard_results, fw)






if __name__ == '__main__':
	main()
