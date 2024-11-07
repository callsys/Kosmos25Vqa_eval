import os
import json
import argparse
import logging

import torch

from transformers.utils import logging as transformers_logging
from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer
)
from transformers.trainer_utils import set_seed

from utils import LayoutLMArguments, LayouyLMTrainingArguments, LayoutLMDataTrainingArguments

from LayoutXLM_local1d.models import (
    LayoutXLM_line1d_Config,
    LayoutXLM_line1d_Tokenizer,
    LayoutXLM_line1d_uncased_Config,
    LayoutXLM_line1d_uncased_Tokenizer,
)
from LayoutXLM_local1d.data.funsd import funsd_dataset, funsd_collator
from LayoutXLM_local1d.metric.funsd_metric import funsd_metrics

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_json", type=str, required=True, help="Config file for HfArgumentParser")
    argparser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = argparser.parse_args()

    while len(logging.root.handlers):
        logging.root.removeHandler(logging.root.handlers[0])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - pid %(process)d - function %(funcName)s - line %(lineno)d -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging.getLogger(__name__).setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    transformers_logging._default_handler.setFormatter(logging.root.handlers[-1].formatter)
    transformers_logging.set_verbosity(logging.root.getEffectiveLevel())

    parser = HfArgumentParser((LayoutLMArguments, LayoutLMDataTrainingArguments, LayouyLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=args.config_json)
    model_args.local_rank = args.local_rank
    data_args.local_rank = args.local_rank
    training_args.local_rank = args.local_rank

    training_args.logging_dir = os.path.join(training_args.output_dir, 'runs')

    set_seed(training_args.seed)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logger.warning(
            f"RANK: {torch.distributed.get_rank()}, LOCAL_RANK: {args.local_rank}, WORLD_SIZE: {torch.distributed.get_world_size()}")
    logger.warning("\n" + json.dumps(dict(os.environ), ensure_ascii=True, indent=4))

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=7,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        add_prefix_space=True,
        use_fast=True,
    )
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer."
        )

    train_dataset, valid_dataset, test_dataset = None, None, None
    if training_args.do_train:
        train_dataset = funsd_dataset(
            data_args=data_args,
            tokenizer=tokenizer,
            mode="train",
        )
        valid_dataset = funsd_dataset(
            data_args=data_args,
            tokenizer=tokenizer,
            mode="valid",
        )

    if training_args.do_predict:
        test_dataset = funsd_dataset(
            data_args=data_args,
            tokenizer=tokenizer,
            mode="test",
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    if training_args.do_train:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator= funsd_collator(token_pad_id=tokenizer.pad_token_id,),
            compute_metrics= funsd_metrics(
                examples=valid_dataset.feature,
                id2labels=valid_dataset.id2labels,
            ),
        )

        trainer.train()
        trainer.save_model()
    if training_args.do_predict:
        assert test_dataset != None
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=funsd_collator(
                token_pad_id=tokenizer.pad_token_id,
            ),
            compute_metrics=funsd_metrics(
                examples=test_dataset.feature,
                id2labels=test_dataset.id2labels,
            ),
        )
        metrics = trainer.evaluate()


if __name__ == '__main__':
    main()