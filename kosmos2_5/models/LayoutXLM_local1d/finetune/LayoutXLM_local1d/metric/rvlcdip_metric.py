import numpy as np
from datasets import load_metric

class rvlcdip_metrics():
    def __init__(
            self,
            examples,
    ):
        self.examples = examples
        self.metric = load_metric("accuracy")


    def __call__(self, result):
        out_label_ids, predictions = result.label_ids, result.predictions
        predictions = np.argmax(predictions, axis=-1)

        labels = out_label_ids.tolist()
        preds = predictions.tolist()

        results = self.metric.compute(predictions=preds, references=labels)

        return results