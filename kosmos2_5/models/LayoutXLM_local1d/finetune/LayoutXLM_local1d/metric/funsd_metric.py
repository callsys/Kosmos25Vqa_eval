import numpy as np
from datasets import load_metric

class funsd_metrics():
    def __init__(
            self,
            examples,
            id2labels,
            pad_token_label_id=-100,
    ):
        self.examples = examples
        self.id2labels = id2labels
        self.pad_token_label_id = pad_token_label_id
        self.metric = load_metric("seqeval")


    def __call__(self, result):
        out_label_ids, predictions = result.label_ids, result.predictions
        predictions = np.argmax(predictions, axis=-1)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    gt_label = self.id2labels[out_label_ids[i, j]]
                    pred_label = self.id2labels[predictions[i, j]]

                    out_label_list[i].append(gt_label)
                    preds_list[i].append(pred_label)

        labels = out_label_list
        preds = preds_list

        results = {}
        cur_results = self.metric.compute(predictions=preds, references=labels)
        results["p"] = cur_results['overall_precision']
        results["r"] = cur_results['overall_recall']
        results["f1"] = cur_results['overall_f1']

        return results