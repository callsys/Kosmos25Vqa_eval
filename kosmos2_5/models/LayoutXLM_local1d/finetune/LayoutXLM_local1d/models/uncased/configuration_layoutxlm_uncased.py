# coding=utf-8
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LAYOUTLMCASED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
    "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
    "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
    "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
}


class LayoutXLM_line1d_uncased_Config(BertConfig):
    model_type = "layoutxlm_line1d_uncased"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        layout_embedding_type="v1",
        coordinate_size=None,
        shape_size=None,
        has_relative_attention_bias=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        has_spatial_attention_bias=False,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        **kwargs
    ):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.layout_embedding_type = layout_embedding_type
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
