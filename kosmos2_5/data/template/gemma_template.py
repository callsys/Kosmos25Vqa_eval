import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from packaging import version

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
import tokenizers
    
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('gemma')
@dataclass
class GemmaTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<eos>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<eos>'])

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1 # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

    @classmethod
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        # separate prompt by special tokens
        patterns = [r'(<image>)', r'(<bbox>.*?</bbox>)', r'(<ocr>.*?</ocr>)', r'(<md>.*?</md>)']
        separators = []
        for pattern in patterns:
            matches = re.findall(pattern, prompt)
            separators.extend(matches)
        full_pattern = '|'.join(map(re.escape, separators))
        prompt_chunks = re.split('(' + full_pattern + ')', prompt)
        prompt_chunks = [chunk for chunk in prompt_chunks if len(chunk)!=0]

        # tokenize prompt chunks
        input_ids = [tokenizer.bos_token_id]
        for i, chunk in enumerate(prompt_chunks):
            if chunk == "<image>":
                chunk_input_ids = [image_token_index]
            elif chunk.startswith("<ocr>") and chunk.endswith("</ocr>"):
                ocr_st_token = tokenizer("<ocr>").input_ids[1]
                ocr_ed_token = tokenizer("</ocr>").input_ids[1]
                chunk_input_ids = tokenizer(chunk.replace("<ocr>", "").replace("</ocr>", "")).input_ids[1:]
                chunk_input_ids = [ocr_st_token] + chunk_input_ids + [ocr_ed_token]
            elif chunk.startswith("<md>") and chunk.endswith("</md>"):
                md_st_token = tokenizer("<md>").input_ids[1]
                md_ed_token = tokenizer("</md>").input_ids[1]
                chunk_input_ids = tokenizer(chunk.replace("<md>", "").replace("</md>", "")).input_ids[1:]
                chunk_input_ids = [md_st_token] + chunk_input_ids + [md_ed_token]
            elif chunk.startswith("<bbox>") and chunk.endswith("</bbox>"):
                pattern = r'(<.*?>)'
                matches = re.findall(pattern, chunk)
                chunk_input_ids = [tokenizer(match).input_ids[-1] for match in matches]
                # chunk_input_ids = tokenizer(chunk).input_ids[1:]
            elif chunk == "":
                chunk_input_ids = []
            else:
                chunk_input_ids = tokenizer(chunk).input_ids[1:]
            input_ids.extend(chunk_input_ids)

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
