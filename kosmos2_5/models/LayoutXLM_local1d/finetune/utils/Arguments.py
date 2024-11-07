from typing import Optional

from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class LayoutLMArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or {layoutlmV1, }"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # model_revision: str = field(
    #     default="main",
    #     metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    # )
    # auth_token: str = field(
    #     default="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",
    #     metadata={
    #         "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
    #                 "with private models)."
    #     },
    # )

@dataclass
class LayouyLMTrainingArguments(TrainingArguments):
    # back_propagate_positive_chunks_only: bool = field(
    #     default=False
    # )
    pass

@dataclass
class LayoutLMDataTrainingArguments:
    # task_name: Optional[str] = field(
    #     default="ner", metadata={
    #         "help": "The name of the task (ner, pos...).",
    #         "choices": ["ner"],
    #     },
    # )
    # dataset_name: str = field(
    #     default=None,
    #     metadata={
    #         "help": "",
    #         "choices": ["funsd"],
    #     },
    # )
    # has_img:bool = field(
    #     default=False
    # )
    # data_download_cache_dir: str = field(
    #     default=None,
    # )
    data_dir: str = field(
        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
    )
    # train_file: str = field(
    #     default=None, metadata={"help": "the filename of training set"}
    # )
    # valid_file: str = field(
    #     default=None, metadata={"help": "the filename of valid set"}
    # )
    # test_file: str = field(
    #     default=None, metadata={"help": "the filename of test set"}
    # )
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # share_textline_bbox: bool = field(
    #     default=False
    # )
    # use_line_1d: bool = field(
    #     default=False
    # )
    # overwrite_cache: bool = field(
    #     default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    # )
    # overwrite_data_download_cache: bool = field(
    #     default=False, metadata={"help": "Overwrite the data cached training and evaluation sets"}
    # )