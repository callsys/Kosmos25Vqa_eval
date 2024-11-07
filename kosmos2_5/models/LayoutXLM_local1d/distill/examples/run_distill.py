#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch 

import transformers
from layoutlmft.data.data_collator import DataCollatorForMaskedVisualLanguageModeling
# from layoutlmft.data.datasets.cdip import CdipDataset
from layoutlmft.data.datasets.mpdfs_full import mpdfsFullDataset
# from layoutlmft.models.layoutlmcased import LayoutLMCasedConfig, LayoutLMCasedForMaskedLM, LayoutLMCasedTokenizer
from transformers import HfArgumentParser, TrainingArguments, set_seed
from Trainer.DistillTrainer import DistillTrainer as Trainer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers import XLMRobertaForMaskedLM
from layoutlmft.models.layoutxlm import LayoutXLMConfig, LayoutXLMForMaskedLM, LayoutXLMTokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.5")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    tea_model_weight: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    stu_model_weight: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    stu_hidden_size: Optional[int] = field(default=384)
    stu_intermediate_size: Optional[int] = field(default=1536)
    stu_num_attention_heads: Optional[int] = field(default=12)
    stu_num_hidden_layers: Optional[int] = field(default=6)

    model_type: Optional[str] = field(
        default=None,
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    layout_embedding_type: str = field(default="v1")
    layout_embedding_v2_coordinate_size: Optional[int] = field(default=128)
    layout_embedding_v2_shape_size: Optional[int] = field(default=128)
    stu_layout_embedding_v2_coordinate_size: Optional[int] = field(default=128)
    stu_layout_embedding_v2_shape_size: Optional[int] = field(default=128)

    def __post_init__(self):
        if self.layout_embedding_type != "v2":
            self.layout_embedding_v2_coordinate_size = None
            self.layout_embedding_v2_shape_size = None


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(default=None)
    dataset_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    segment_layout_embedding: bool = field(default=True)
    skip_sample_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )

def load_xlmR_weight(model, stu_model_weight):
    model_state = model.state_dict()
    # xlmr_state = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large").state_dict()
    xlmr_state = torch.load(stu_model_weight)


    complete_flag = {}
    for key, val in xlmr_state.items():
        new_key = key.replace("roberta.", "layoutlmcased.")

        assert new_key not in complete_flag.keys()

        if new_key in [
            'layoutlmcased.embeddings.word_embeddings.weight',
            'lm_head.decoder.weight',
        ]:
            model_state[new_key][:val.shape[0], :] = val
        elif new_key in [
            'lm_head.bias',
            'lm_head.decoder.bias',
        ]:
            continue
            # model_state[new_key][:val.shape[0]] = val
        else:
            try:
                assert val.shape == model_state[new_key].shape
                model_state[new_key] = val
            except:
                continue
        complete_flag[new_key] = 1
    model.load_state_dict(model_state, strict=True)


    # model.load_state_dict(torch.load("/mnt/conversationhub/projects/layoutxlm/output/layoutxlm_large_exclude1d_400m_bs_4_gas_2_lr_5e-5_wr_0.003_ms_3000000_pt_clean/checkpoint-590000/pytorch_model.bin", map_location=torch.device('cpu')), strict=True)
    return model


def load_teacher(model_args):
    config = LayoutXLMConfig.from_pretrained(
        "xlm-roberta-large",
        layout_embedding_type=model_args.layout_embedding_type,
        coordinate_size=model_args.layout_embedding_v2_coordinate_size,
        shape_size=model_args.layout_embedding_v2_shape_size,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        vocab_size=250008,
    )
    tokenizer = LayoutXLMTokenizer.from_pretrained("xlm-roberta-large")
    model = LayoutXLMForMaskedLM(config=config)
    model.load_state_dict(torch.load(
        model_args.tea_model_weight,
        map_location=torch.device('cpu')), strict=False)
    return config, tokenizer, model

def load_student(model_args):
    config = LayoutXLMConfig.from_pretrained(
        "xlm-roberta-large",
        layout_embedding_type=model_args.layout_embedding_type,
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        vocab_size=250008,
        hidden_size=model_args.stu_hidden_size,
        intermediate_size=model_args.stu_intermediate_size,
        num_attention_heads=model_args.stu_num_attention_heads,
        num_hidden_layers=model_args.stu_num_hidden_layers,
    )
    model = LayoutXLMForMaskedLM(config=config)
    model = load_xlmR_weight(model, model_args.stu_model_weight)

#    model.load_state_dict(torch.load("/mnt/output/projects/layoutxlm/output/layoutxlm_small_12384_local1d_400m_bs_8_gas_8_lr_1e-3_wr_0.01_ms_1000000_xlmr_small_init_distill_fp16_wus/checkpoint-140000/pytorch_model.bin"), strict=True)

    return config, None, model



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.seed = training_args.seed
    data_args.local_rank = training_args.local_rank

    # Detecting last checkpoint.
    last_checkpoint = None
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        if os.path.exists(training_args.output_dir) == False:
            os.makedirs(training_args.output_dir)
        transformers.utils.logging.add_handler(
            logging.FileHandler(os.path.join(training_args.output_dir, "train.log"), mode="w")
        )
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tea_config, tokenizer, tea_model = load_teacher(model_args)
    stu_config, _, stu_model = load_student(model_args)

    # print('w')
    # config = LayoutXLMConfig.from_pretrained(
    #     "xlm-roberta-large",
    #     layout_embedding_type=model_args.layout_embedding_type,
    #     coordinate_size=model_args.layout_embedding_v2_coordinate_size,
    #     shape_size=model_args.layout_embedding_v2_shape_size,
    #     has_relative_attention_bias=True,
    #     has_spatial_attention_bias=True,
    #     vocab_size=250008,
    # )
    #
    # tokenizer = LayoutXLMTokenizer.from_pretrained("xlm-roberta-large")
    # model = LayoutXLMForMaskedLM(config=config)
    # model = load_xlmR_weight(model)

    if training_args.do_train:
        train_dataset = mpdfsFullDataset(data_args, tokenizer)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForMaskedVisualLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
    )

    n_parameters = sum(p.numel() for p in stu_model.parameters() if p.requires_grad)
    print("Model = %s" % str(stu_model))
    print("number of params:", n_parameters)

    # Initialize our Trainer
    trainer = Trainer(
        tea_model=tea_model,
        model=stu_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
#        checkpoint = None
#        if training_args.resume_from_checkpoint is not None:
#            checkpoint = training_args.resume_from_checkpoint
#        elif last_checkpoint is not None:
#            checkpoint = last_checkpoint
#        train_result = trainer.train(resume_from_checkpoint=True)
        train_result = trainer.train(resume_from_checkpoint=False)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
