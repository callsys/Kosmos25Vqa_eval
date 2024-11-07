# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RoBERTa."""

from transformers.models.roberta import RobertaTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/vocab.json",
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/vocab.json",
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/vocab.json",
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/vocab.json",
        "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/vocab.json",
    },
    "merges_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/merges.txt",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/merges.txt",
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/merges.txt",
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/merges.txt",
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/merges.txt",
        "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}


class LayoutLMCasedTokenizer(RobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
