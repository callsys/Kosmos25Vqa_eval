
import logging
import os

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    FairseqDataset,
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    RawLabelDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq import utils
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
from fairseq.data import Dictionary, data_utils
from omegaconf import II
from fairseq import metrics, search, tokenizer, utils
from kosmos2_5.data.utils import SPECIAL_SYMBOLS

logger = logging.getLogger(__name__)
MAX_PATHES=4096

@dataclass
class GenerationConfig(LanguageModelingConfig):
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")
    dict_path: str = field(
        default="",
        metadata={
            "help": "dictionary path"
        },
    )
    image_feature_length: int = field(
        default=0,
        metadata={
            "help": "image feature length"
        },
    )   

class customDataset(FairseqDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        try:
            return torch.stack(samples)
        except:
            return samples

class RawImageDataset(FairseqDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.stack(samples)


@register_task("generation", dataclass=GenerationConfig)
class GenerationTask(LanguageModelingTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        if len(args.dict_path) > 0:
            dictionary = Dictionary.load(args.dict_path)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        dictionary.add_symbol("<mask>")
        for special_symbol in SPECIAL_SYMBOLS:
            dictionary.add_symbol(special_symbol)
        
        dictionary.pad_to_multiple_(args.required_batch_size_multiple)
        output_dictionary = dictionary
        logger.info("dictionary: {} types".format(len(dictionary)))
        return (dictionary, output_dictionary)

    def build_dataset_for_caption_inference(self, src_tokens, src_lengths, img_src_tokens, img_gpt_input_mask, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        img_gpt_input_mask = StripTokenDataset(
            TokenBlockDataset(
                img_gpt_input_mask,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = dataset
        # PrependTokenDataset(
        #     dataset,
        #     token=(
        #         self.source_dictionary.bos()
        #         if getattr(self.args, "add_bos_token", False)
        #         else self.source_dictionary.eos()
        #     ),
        # )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'img_src_tokens': RawImageDataset(
                        img_src_tokens,
                    ),
                    'img_gpt_input_mask': PadDataset(
                        img_gpt_input_mask,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def build_dataset_for_caption_inference_with_embed(self, src_tokens, src_lengths, img_src_tokens, img_attention_masks, img_gpt_input_mask, segment_tokens, image_paths, widths, heights, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        img_gpt_input_mask = StripTokenDataset(
            TokenBlockDataset(
                img_gpt_input_mask,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        img_attention_masks = StripTokenDataset(
            TokenBlockDataset(
                img_attention_masks,
                [MAX_PATHES],
                block_size=None,  # ignored for "eos" break mode
                pad=0,
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        # chunk_tokens = StripTokenDataset(
        #     TokenBlockDataset(
        #         chunk_tokens,
        #         src_lengths,
        #         block_size=None,  # ignored for "eos" break mode
        #         pad=self.source_dictionary.pad(),
        #         eos=self.source_dictionary.eos(),
        #         break_mode="eos",
        #     ),
        #     # remove eos from (end of) target sequence
        #     self.source_dictionary.eos(),
        # )
        segment_tokens = StripTokenDataset(
            TokenBlockDataset(
                segment_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = dataset
        # PrependTokenDataset(
        #     dataset,
        #     token=(
        #         self.source_dictionary.bos()
        #         if getattr(self.args, "add_bos_token", False)
        #         else self.source_dictionary.eos()
        #     ),
        # )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'img_src_tokens': RawImageDataset(
                        img_src_tokens,
                    ),
                    'img_gpt_input_mask': PadDataset(
                        img_gpt_input_mask,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    'img_attention_masks': PadDataset(
                        img_attention_masks,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    # 'chunk_tokens': PadDataset(
                    #     chunk_tokens,
                    #     pad_idx=0,
                    #     left_pad=False,
                    # ),
                    'segment_tokens': PadDataset(
                        segment_tokens,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    "image_paths": customDataset(
                        image_paths,
                    ),
                    'widths': customDataset(
                        widths,
                    ),
                    'heights': customDataset(
                        heights,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def build_dataset_for_speech_inference(self, src_tokens, src_lengths, aud_src_tokens, aud_gpt_input_mask, audio_masks, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        aud_gpt_input_mask = StripTokenDataset(
            TokenBlockDataset(
                aud_gpt_input_mask,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = dataset
        # PrependTokenDataset(
        #     dataset,
        #     token=(
        #         self.source_dictionary.bos()
        #         if getattr(self.args, "add_bos_token", False)
        #         else self.source_dictionary.eos()
        #     ),
        # )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'aud_src_tokens': RawImageDataset(
                        aud_src_tokens,
                    ),
                    'aud_gpt_input_mask': PadDataset(
                        aud_gpt_input_mask,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    'aud_masks': RawImageDataset(
                        audio_masks,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    # def build_generator(
    #     self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    # ):
    #     if getattr(args, "score_reference", False):
    #         from fairseq.sequence_scorer import SequenceScorer
    #
    #         return SequenceScorer(
    #             self.target_dictionary,
    #             compute_alignment=getattr(args, "print_alignment", False),
    #             compute_vocab_dist=getattr(args, "compute_vocab_dist", False)
    #         )
    #
    #     from fairseq.sequence_generator import (
    #         SequenceGenerator,
    #         SequenceGeneratorWithAlignment,
    #     )
    #     try:
    #         from fairseq.fb_sequence_generator import FBSequenceGenerator
    #     except ModuleNotFoundError:
    #         pass
    #
    #     # Choose search strategy. Defaults to Beam Search.
    #     sampling = getattr(args, "sampling", False)
    #     sampling_topk = getattr(args, "sampling_topk", -1)
    #     sampling_topp = getattr(args, "sampling_topp", -1.0)
    #     diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
    #     diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
    #     match_source_len = getattr(args, "match_source_len", False)
    #     diversity_rate = getattr(args, "diversity_rate", -1)
    #     constrained = getattr(args, "constraints", False)
    #     prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
    #     if (
    #         sum(
    #             int(cond)
    #             for cond in [
    #                 sampling,
    #                 diverse_beam_groups > 0,
    #                 match_source_len,
    #                 diversity_rate > 0,
    #             ]
    #         )
    #         > 1
    #     ):
    #         raise ValueError("Provided Search parameters are mutually exclusive.")
    #     assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
    #     assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"
    #
    #     if sampling:
    #         search_strategy = search.Sampling(
    #             self.target_dictionary, sampling_topk, sampling_topp
    #         )
    #     elif diverse_beam_groups > 0:
    #         search_strategy = search.DiverseBeamSearch(
    #             self.target_dictionary, diverse_beam_groups, diverse_beam_strength
    #         )
    #     elif match_source_len:
    #         # this is useful for tagging applications where the output
    #         # length should match the input length, so we hardcode the
    #         # length constraints for simplicity
    #         search_strategy = search.LengthConstrainedBeamSearch(
    #             self.target_dictionary,
    #             min_len_a=1,
    #             min_len_b=0,
    #             max_len_a=1,
    #             max_len_b=0,
    #         )
    #     elif diversity_rate > -1:
    #         search_strategy = search.DiverseSiblingsSearch(
    #             self.target_dictionary, diversity_rate
    #         )
    #     elif constrained:
    #         search_strategy = search.LexicallyConstrainedBeamSearch(
    #             self.target_dictionary, args.constraints
    #         )
    #     elif prefix_allowed_tokens_fn:
    #         search_strategy = search.PrefixConstrainedBeamSearch(
    #             self.target_dictionary, prefix_allowed_tokens_fn
    #         )
    #     else:
    #         search_strategy = search.BeamSearch(self.target_dictionary)
    #
    #     extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
    #     if seq_gen_cls is None:
    #         if getattr(args, "print_alignment", False):
    #             seq_gen_cls = SequenceGeneratorWithAlignment
    #             extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
    #         elif getattr(args, "fb_seq_gen", False):
    #             seq_gen_cls = FBSequenceGenerator
    #         else:
    #             seq_gen_cls = SequenceGenerator
    #
    #     return seq_gen_cls(
    #         models,
    #         self.target_dictionary,
    #         beam_size=getattr(args, "beam", 5),
    #         max_len_a=getattr(args, "max_len_a", 0),
    #         max_len_b=getattr(args, "max_len_b", 200),
    #         min_len=getattr(args, "min_len", 1),
    #         normalize_scores=(not getattr(args, "unnormalized", False)),
    #         len_penalty=getattr(args, "lenpen", 1),
    #         unk_penalty=getattr(args, "unkpen", 0),
    #         temperature=getattr(args, "temperature", 1.0),
    #         match_source_len=getattr(args, "match_source_len", False),
    #         no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
    #         search_strategy=search_strategy,
    #         **extra_gen_cls_kwargs,
    #     )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                # if prefix_tokens[:, 0].eq(bos_token).all():
                #     prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )