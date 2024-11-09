import random
import time
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import openai
import json
import os
import tqdm
import argparse
import textdistance as td

# init the OpenAI client
API_INFOS = [
    {
        "endpoints": "https://conversationhubeastus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubeastus2.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubwestus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubwestus3.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readineastus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readineastus2.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readinnorthcentralus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readinsouthcentralus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readinwestus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readinwestus3.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
]

class Openai():
    def __init__(
            self,
            apis,
            identity_id="8dfc08a3-fbaa-4aa2-b776-a510cd4ef0b5",

    ):
        self.identity_id = identity_id
        flag = True
        while flag:
            try:
                self.token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(managed_identity_client_id=self.identity_id),
                    "https://cognitiveservices.azure.com/.default"
                )
                flag = False
                break
            except:
                continue

        self.clients_weight = [apis[i]['speed'] for i in range(len(apis))]
        weight_sum = sum(self.clients_weight)
        for i in range(len(self.clients_weight)):
            self.clients_weight[i] /= weight_sum


        selected_api = random.choices(apis, weights=self.clients_weight, k=1)[0]

        self.client = AzureOpenAI(
            azure_endpoint=selected_api['endpoints'],
            azure_ad_token_provider=self.token_provider,
            api_version="2024-04-01-preview",
            max_retries=0,
        )
        self.model = selected_api['model']


    def call(self, content, client_index = None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"{content}\n"},
            ]
             },
        ]

        client = self.client
        model = self.model

        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=64,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )

                # client.chat.completions.with_raw_response
                results = completion.choices[0].message.content
                return results
            except openai.RateLimitError as e:
                time.sleep(1)
            except Exception as e:
                print(e)
                cur_retry += 1
        return ""


# prepare the data for testing
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--shard", type=str, default="1/4")
	parser.add_argument("--num-sample", type=int, default=-1)
	args = parser.parse_args()
	return args

# evaluation metrics
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

# docvqa, question, answer
if __name__ == '__main__':
    args = get_args()
    
    # prepare data
    vqa_data = json.load(open("/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/vqa/test.json"))
    data_dir = "/mnt/msranlp/yuzhongzhao/docvqa_kosmos/DocVQA/dataset/kosmos_d/"
    samples = vqa_data[0]['source']

    shard = args.shard
    cur_shard, num_shard = int(shard.split("/")[0]) - 1, int(shard.split("/")[1])
    if args.num_sample > 0:
        samples = samples[:args.num_sample]
    samples = samples[cur_shard::num_shard]

    # init the OpenAI client based on shard
    oai_clients = Openai(
        apis=[API_INFOS[cur_shard % len(API_INFOS)]]
    )

    # evaluation
    gt_pred_pairs = []
    shard_results = dict()

    for idx, sample in enumerate(tqdm.tqdm(samples)):
        # extract the MSOCR result
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

        # to be fixed
        question = "Based on the given OCR result, please answer the question given by human.\n" + doc_str + question
        
        res = oai_clients.call(question)
        # to be fixed
        pred_answer = res.split("\n")[-1].strip()
        
        gt_pred_pairs.append([answers, pred_answer])
        questionId = doc_str["extra_info"]["questionId"]
        shard_results[questionId] = {"pred_answer": pred_answer, "answers": answers}

        if (idx+1) % 20 == 0:
            preds = [el[1] for el in gt_pred_pairs]
            gts = [el[0] for el in gt_pred_pairs]
            _, anls = anls_metric_str(preds, gts)
            print(f"\nANLS score at evaluation step ({idx+1}/{len(samples)}) : {anls}\n")

    # 存在当前目录下
    shard_result_dir = "gpt4o_docvqa_results"
    save_result_path = os.path.join(f"gpt4o_docvqa_{cur_shard}_{num_shard}.json")
    print(f"dump results to {save_result_path}")
    with open(save_result_path, "w") as fw:
        json.dump(shard_results, fw)