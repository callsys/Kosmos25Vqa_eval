'''
	inference code from: kosmos2.5
	anls metric code from: https://github.com/allanj/LayoutLMv3-DocVQA/blob/master/src/utils.py
'''
import json
import tqdm
import argparse
import os
import textdistance as td

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt", "-c", type=str, default="/mnt/msranlp/yuzhongzhao/ckpts/vqa529/checkpoint_1_210500.pt")
	parser.add_argument("--num-sample", type=int, default=-1)
	parser.add_argument("--shard", type=str, default="8")
	args = parser.parse_args()
	return args

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

	num_shard = int(args.shard)

	# shard_result_dir = "./docvqa_results"
	shard_result_dir = os.path.basename(os.path.basename(args.ckpt)) + f"_{args.num_sample}"
	# shard_result_dir = os.path.basename(os.path.basename(args.ckpt))
	assert os.path.exists(shard_result_dir)
	save_result_paths = [os.path.join(shard_result_dir, f"evaluate_docvqa_{cur_shard}_{num_shard}.json") for cur_shard in range(num_shard)]

	gather_results = dict()

	for result_path in tqdm.tqdm(save_result_paths):
		assert os.path.exists(result_path)

		with open(result_path, "r") as fr:
			json_file = json.load(fr)

		gather_results.update(json_file)

	values = list(gather_results.values())
	preds = [el["pred_answer"].strip() for el in values]
	gts = [el["answers"] for el in values]
	assert len(preds) == len(gts)
	_, anls = anls_metric_str(preds, gts)
	print(f"\nANLS score of {len(preds)} samples : {anls}\n")


if __name__ == '__main__':
	main()
