from dataclasses import dataclass, field
from typing import Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

import logging
import numpy as np
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.data import Dictionary
# from fairseq.utils import safe_getattr, safe_hasattr
from omegaconf import II

from fairseq.modules import LayerNorm
from fairseq.models import (
  BaseFairseqModel,
  register_model,
  register_model_architecture,
)
from fairseq.models.roberta import (
    roberta_large_architecture,
    roberta_base_architecture,
    RobertaEncoder,
    RobertaModel,
)
from fairseq.models.transformer_lm import (
  TransformerLanguageModelConfig,
  TransformerLanguageModel,
  base_gpt3_architecture,
)
from kosmos2_5.models.connector import build_connector
from kosmos2_5.models.gpt import GPTModelConfig

from torchscale.architecture.config import EncoderConfig
# from torchscale.model.BEiT3 import BEiT3

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from kosmos2_5.models.LayoutXLM_local1d.finetune.LayoutXLM_local1d.models.uncased.modeling_layoutxlm_uncased import BertPreTrainedModel, BertModel, LayoutXLM_line1d_uncased_Config
from kosmos2_5.models.gpt import Decoder, FairseqIncrementalDecoder, PositionalEmbedding, Tensor, Embedding, TextEmbedding, DEFAULT_MAX_TARGET_POSITIONS, distributed_utils, DecoderConfig

logger = logging.getLogger(__name__)

def slice_tokens_for_mlm(A, indx, num_elem=2):
    all_indx = indx[:,None] + torch.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


class LayoutXLM_uncased_ForGPT(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        bbox=None,
        position_ids=None,
        rel_pos_mat=None,
        segment_ids=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            rel_pos_mat=rel_pos_mat,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        return sequence_output


AutoModelForTokenClassification.register(LayoutXLM_line1d_uncased_Config, LayoutXLM_uncased_ForGPT)


class LMDecoder(Decoder, FairseqIncrementalDecoder):
    def forward(self, src_tokens, **kwargs):
        self_attn_padding_mask = src_tokens.eq(self.dictionary.pad())
        return super().forward(src_tokens, self_attn_padding_mask, **kwargs)

    def max_positions(self):
        return self.embed_positions.max_positions

    def reorder_incremental_state_scripting(
        self,
        incremental_state,
        new_order,
    ):
        for module in incremental_state:
            for key in incremental_state[module]:
                result = incremental_state[module][key].index_select(0, new_order)
                incremental_state[module][key] = result

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
        first_step: bool = False,
        mlm_features: Optional[Tensor] = None,
        gpt_input_mask: Optional[Tensor] = None,
        img_features: Optional[Tensor] = None,
        img_gpt_input_mask: Optional[Tensor] = None,
        layoutlm_features: Optional[Tensor] = None,
        layoutlm_gpt_input_mask: Optional[Tensor] = None,
        aud_features: Optional[Tensor] = None,
        aud_gpt_input_mask: Optional[Tensor] = None,
        chunk_tokens: Optional[Tensor] = None,
        segment_tokens: Optional[Tensor] = None,
    ):
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state
            )
            if self.segment_emb is not None:
                segment_emb = self.segment_emb(segment_tokens)
                positions = positions + segment_emb

        if incremental_state is not None and not first_step:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        gpt_embed_output = token_embedding
        if img_features is not None:
            gpt_embed_output[img_gpt_input_mask] = img_features
        if layoutlm_features is not None:
            gpt_embed_output[layoutlm_gpt_input_mask] = layoutlm_features

        x = embed = self.embed_scale * gpt_embed_output

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def forward(
        self,
        prev_output_tokens,
        self_attn_padding_mask=None,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        first_step=False,
        **kwargs
    ):
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state, first_step=first_step, **kwargs
        )

        inner_states = [x]

        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or first_step:
                self_attn_mask = None
                if first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                self_attn_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None
                else None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                # self_attn_rel_pos=self_attn_rel_pos_bias,
                # cross_attn_rel_pos=cross_attn_rel_pos_bias,
                self_attn_rel_pos=None,
                cross_attn_rel_pos=None,
                first_step=first_step,
                # first_step=True,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }


class GPTmodel(TransformerLanguageModel):

    @classmethod
    def build_model(cls, args, task):
        model = TransformerLanguageModel.build_model(args, task)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )

        embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                args.decoder_embed_dim,
                task.dictionary.pad(),
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, len(task.dictionary), bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )

        if getattr(args, "moe_freq", 0) > 0 and (
            getattr(args, "fp16", False)
            and not getattr(args, "memory_efficient_fp16", False)
            and getattr(args, "ddp_backend", None) != "fully_sharded"
        ):
            assert (
                args.fp16_no_flatten_grads
            ), "If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm"

        args.ddp_rank = distributed_utils.get_data_parallel_rank()

        config = DecoderConfig()
        config.override(args)

        decoder = LMDecoder(
            config,
            embed_tokens,
            embed_positions,
            output_projection,
            is_encoder_decoder=False,
            dictionary=task.dictionary,
        )
        decoder.chunk_emb = None
        decoder.segment_emb = None
        if args.segment_emb:
            decoder.segment_emb = TextEmbedding(2, args.decoder_embed_dim)
        model.decoder = decoder
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(len(dictionary), embed_dim, dictionary.pad())


@dataclass
class DocGPTModelConfig(GPTModelConfig):
    pass


@register_model("docgptmodel", dataclass=DocGPTModelConfig)
class DocGPTmodel(BaseFairseqModel):

    def __init__(
            self,
            args,
            gpt_model,
            img_model=None,
            layoutlm_model=None,
            img_connector=None,
            layoutlm_connector=None,
            bos=0, eos=2):
        """
        text_model: bidirectional text model, such as roberta, bert, electra
        img_model: image model, such as ViT, CLIP, BEIT
        aud_model: audio model, such as HuBERT, wavLM
        """
        super().__init__()
        self.args = args
        self.gpt_model = gpt_model

        self.img_model = img_model
        self.layoutlm_model = layoutlm_model
        self.img_connector = img_connector
        self.layoutlm_connector = layoutlm_connector
        self.bos = bos
        self.eos = eos
        self.classification_heads = nn.ModuleDict()
        self.ft_type = args.ft_type

    @classmethod
    def build_model(cls, args, task):
        if hasattr(task, "all_dict"):
            task.dictionary = task.all_dict
        gpt_model = GPTmodel.build_model(args, task)
        logger.info("gpt args is {}".format(args))

        img_model, img_connector = cls.load_image_model(args, task)
        layoutlm_model, layoutlm_connector = cls.load_layoutlm_model(args, task)

        model = cls(args, gpt_model,
                    # text_model=text_model, text_connector=text_connector,
                    img_model=img_model, img_connector=img_connector,
                    layoutlm_model=layoutlm_model, layoutlm_connector=layoutlm_connector,
                    # aud_model=aud_model, aud_connector=aud_connector,
                    bos=task.dictionary.bos_index,
                    eos=task.dictionary.eos_index)

        return model

    def forward(self, src_tokens,
                mlm_src_tokens=None, gpt_input_mask=None,
                img_src_tokens=None, img_gpt_input_mask=None,
                layoutlm_input=None, layoutlm_gpt_input_mask=None,
                aud_src_tokens=None, aud_gpt_input_mask=None,
                gpt_loss_mask=None, mlm_mask=None, classification_head_name=None, **kwargs):

        if classification_head_name is None:
            if mlm_src_tokens is not None:
                # mlm
                mlm_output, _ = self.text_model(mlm_src_tokens, features_only=True)
                mlm_output = mlm_output[mlm_mask]
                if self.text_connector is not None:
                    # linear projection layer
                    mlm_output = self.text_connector(mlm_output)
            else:
                mlm_output = None

            if img_src_tokens is not None:
                img_output = self.get_image_representation(img_src_tokens)
            else:
                img_output = None

            if layoutlm_input is not None:
                layoutlm_output = self.get_layoutlm_representation(layoutlm_input)
            else:
                layoutlm_output = None

            if aud_src_tokens is not None:
                aud_output = self.get_audio_representation(aud_src_tokens, kwargs['aud_mask'])
            else:
                aud_output = None

            # gpt
            x, extra = self.gpt_model(src_tokens,
                                      mlm_features=mlm_output, gpt_input_mask=gpt_input_mask,
                                      img_features=img_output, img_gpt_input_mask=img_gpt_input_mask,
                                      layoutlm_features=layoutlm_output,
                                      layoutlm_gpt_input_mask=layoutlm_gpt_input_mask,
                                      aud_features=aud_output, aud_gpt_input_mask=aud_gpt_input_mask,
                                      **kwargs)

            # loss mask
            extra["loss_mask"] = gpt_loss_mask
            return x, extra

    def get_image_representation(self, img_src_tokens, image_attention_masks):
        # image
        img_output = self.img_model(img_src_tokens, image_attention_masks)
        img_output = F.normalize(img_output[0], dim=-1)
        src_len = img_output.size(1)
        img_output = img_output.reshape(-1, img_output.size(-1))
        if self.img_connector is not None:
            img_output = self.img_connector(img_output, src_len=src_len)
        return img_output

    def get_layoutlm_representation(self, layoutlm_input):
        layoutlm_output = self.layoutlm_model(**layoutlm_input)
        src_len = layoutlm_output.size(1)
        if self.layoutlm_connector is not None:
            # linear projection layer
            layoutlm_output = self.layoutlm_connector(layoutlm_output, src_len=src_len)
        return layoutlm_output

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def supported_targets(self):
        return {"future"}

    @classmethod
    def load_image_model(cls, args, task):
        from transformers import Pix2StructVisionModel
        model = Pix2StructVisionModel.from_pretrained(args.image_encoder)
        connector = build_connector(args, model.config.hidden_size, args.decoder_embed_dim)

        return model, connector

    @classmethod
    def load_layoutlm_model(cls, args, task):
        model_name_or_path = "/mnt/msranlp/yuzhongzhao/layoutlmv3/"
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=7,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )
        layoutlm_args = copy.deepcopy(args)
        layoutlm_args.connector = "simple"
        connector = build_connector(layoutlm_args, model.config.hidden_size, args.decoder_embed_dim)

        return model, connector


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        ft_type
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.ft_type = ft_type

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("docgptmodel", "docgptmodel_large")
def gptmodel_large(args):
    # 1.3B params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)

    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    base_gpt3_architecture(args)
    roberta_large_architecture(args)

