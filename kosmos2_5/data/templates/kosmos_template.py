import re
from dataclasses import dataclass
import copy
import numpy as np

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter

from kosmos2_5.data.utils import SOB_SYMBOL, EOB_SYMBOL, OCR_SYMBOL, MD_SYMBOL, BOI_SYMBOL, EOI_SYMBOL

# question = "You're a smart document AI assistant. Based on the given image, please answer the question given by human.\n" \
#            "Human :" + question + "\n" + "Assistant :"
    
# system = "You're a smart document AI assistant. Based on the given image, please answer the question given by human."
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@dataclass
class KosmosTemplate(Template):
    # format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    # format_user: "Formatter" = StringFormatter(slot="Human :" + "{{content}}" + "\n")
    # format_assistant: "Formatter" = StringFormatter(slot="Assistant :" + "{{content}}" + "<eos>")
    # system: "Formatter" = EmptyFormatter(slot=system+"\n")
    # separator: "Formatter" = EmptyFormatter(slot=['Assistant :', '<eos>'])
    format_image_token: "Formatter" = StringFormatter(slot="{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<eos>")
    system: "Formatter" = EmptyFormatter(slot="<image><md>" + system + " ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<eos>'])

    def __init__(self, tokenizer, dictionary):
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.boi_id = self.dictionary.index(BOI_SYMBOL)
        self.eoi_id = self.dictionary.index(EOI_SYMBOL)
        self.sob_id = self.dictionary.index(SOB_SYMBOL)
        self.eob_id = self.dictionary.index(EOB_SYMBOL)
        self.md_id = self.dictionary.index(MD_SYMBOL)
        self.ocr_id = self.dictionary.index(OCR_SYMBOL)
        self.bos_id = self.dictionary.bos_index
        self.pad_id = self.dictionary.pad()
        self.img_id = -200

    def encode(self, messages, mode='train', **kwargs):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages)
        prompt = self.prompt(question_list, answer_list)
        text_tokens = self.tokenizer_image_token(prompt, **kwargs)

        # make other kosmos25 inputs
        text_input_mask = np.zeros(len(text_tokens), dtype=np.int32)
        text_input_mask[text_tokens == self.img_id] = 1
        segment_ids = np.zeros(len(text_tokens), dtype=np.int32)
        segment_ids[text_tokens == self.img_id] = 1
        segment_ids[text_tokens == self.boi_id] = 1
        segment_ids[text_tokens == self.eoi_id] = 1
        # reset values that not in the dict to pad
        text_tokens[text_tokens == self.img_id] = self.boi_id

        ret = {
            "text_tokens": text_tokens,
            "text_input_mask": text_input_mask,
            "segment_ids": segment_ids,
            "prompt": prompt
        }

        if mode == 'train':
            label_ret = self.make_labels(text_tokens, prompt, **kwargs)
            ret.update(label_ret)
            return ret
        else:
            return ret

    def encode_org(self, messages, mode='train', **kwargs):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages)

        prompt = "<image><md>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: what is the alcohol percentage? ASSISTANT: 4.5%<eos>"

        question = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: " + question_list[0].replace("<image>\n", "")
        answer = answer_list[0]

        ids = []
        # vqa
        append_eos = True
        fs_dict = self.dictionary
        ids += [self.dictionary.index(MD_SYMBOL)]
        ids += [fs_dict.index(str(token)) for token in self.tokenizer.encode(question)]

        question_ids = copy.deepcopy(ids)

        ids += [self.dictionary.index(str(token)) for token in self.tokenizer.encode(answer)]

        if append_eos:
            ids.append(self.dictionary.eos_index)

        question_tokens = question_ids
        qa_tokens = ids

        qa_length = len(qa_tokens)
        question_length = len(question_tokens)
        answer_length = qa_length - question_length

        text_tokens = [self.bos_id] + [self.boi_id] * (2048 + 1) + [self.eoi_id] + qa_tokens
        text_input_mask = [0] + [0] + [1] * (2048) + [0] + [0] * qa_length
        text_loss_mask = [0] + [0] + [0] * (2048) + [0] + [0] * question_length + [
            1] * answer_length  # calculate loss for answer only
        segment_ids = [0] + [1] + [1] * (2048) + [1] + [0] * qa_length  # image tokens

        text_tokens = np.array(text_tokens)
        text_input_mask = np.array(text_input_mask).astype(np.int32)
        text_loss_mask = np.array(text_loss_mask).astype(np.int32)
        segment_ids = np.array(segment_ids).astype(np.int32)

        ret = {
            "text_tokens": text_tokens,
            "text_input_mask": text_input_mask,
            "segment_ids": segment_ids,
            "text_loss_mask": text_loss_mask,
            "prompt": prompt
        }

        return ret

    def get_list_from_message(self, messages):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])
            else:
                answer_list.append(message['value'])

        assert len(question_list) == len(answer_list), \
            f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list

    def prompt(self, question_list, answer_list):
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if BOI_SYMBOL in question:
                question = question.replace(BOI_SYMBOL, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            msg += self.format_assistant.apply(content=answer)
        return msg

    def tokenizer_image_token(self, prompt, **kwargs):
        text_tokens = [self.bos_id]

        patterns = [r'(<image>)', r'(<bbox>.*?</bbox>)', r'(<ocr>)', r'(<md>)', r'(<eos>)']
        separators = []
        for pattern in patterns:
            matches = re.findall(pattern, prompt)
            separators.extend(matches)
        full_pattern = '|'.join(map(re.escape, separators))
        prompt_chunks = re.split('(' + full_pattern + ')', prompt)
        prompt_chunks = [chunk for chunk in prompt_chunks if len(chunk) != 0]

        for i, chunk in enumerate(prompt_chunks):
            if chunk == "<image>":
                # image tokens
                if kwargs.get("has_image", 0) == 1:
                    image_token_length = kwargs.get("image_token_length", 2048) # 2048 is the default setting of kosmos25
                    chunk_input_ids = [self.boi_id] + [self.img_id] * image_token_length + [self.eoi_id]
                    text_tokens.extend(chunk_input_ids)
                else:
                    pass # no image tokens
            elif chunk.startswith(OCR_SYMBOL):
                text_tokens.append(self.ocr_id)
            elif chunk.startswith(MD_SYMBOL):
                text_tokens.append(self.md_id)
            elif chunk.startswith(SOB_SYMBOL) and chunk.endswith(EOB_SYMBOL):
                # bbox text
                pattern = r'(<.*?>)'
                matches = re.findall(pattern, chunk)
                chunk_input_ids = [self.dictionary.index(token) for token in matches]
                text_tokens.extend(chunk_input_ids)
            elif chunk.startswith("<eos>"):
                text_tokens.append(self.dictionary.eos_index)
            else:
                # normal text
                chunk_input_ids = [self.dictionary.index(str(tok)) for tok in self.tokenizer.encode(chunk)]
                text_tokens.extend(chunk_input_ids)

        return np.array(text_tokens)

    def make_labels(self, text_tokens, prompt, **kwargs):
        text_tokens = np.array(text_tokens)
        text_loss_mask = np.ones(len(text_tokens), dtype=np.int32)
        sep, eos_token = self.separator.apply()
        total_len = (text_tokens != self.pad_id).sum()
        rounds = prompt.split(eos_token)
        text_loss_mask, cur_len = self._make_masks(text_loss_mask, sep, rounds, **kwargs)

        if cur_len != total_len:
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
            )
            print("number of rounds: ", len(rounds) - 1)
            print("rounds: ", rounds[:-1])
            print("prompt: ", prompt)
            print(text_tokens)
            print(text_loss_mask)
            text_loss_mask[:] = 0

        ret = {
            "text_loss_mask": text_loss_mask,
        }

        return ret

    def _make_masks(self, text_loss_mask, sep, rounds, **kwargs):
        cur_len = 1  # bos
        eos_token_length = 1
        bos_token_length = 1
        text_loss_mask[:cur_len] = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, **kwargs)) + eos_token_length - bos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], **kwargs)) - 1 - bos_token_length
            # instruction_len = len(self.tokenizer_image_token(parts[0], **kwargs)) - bos_token_length
            text_loss_mask[cur_len: cur_len + instruction_len] = 0
            cur_len += round_len

        text_loss_mask[cur_len:] = 0
        return text_loss_mask, cur_len
