from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoModelForSequenceClassification,AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, XLMRobertaConverter

from .configuration_layoutxlm_line1d import LayoutXLM_line1d_Config
from .modeling_layoutxlm_line1d import (
    LayoutLMCasedModel,
    LayoutXLM_line1d_ForTokenClassification,
    LayoutXLM_line1d_ForSequenceClassification,
)
from .tokenization_layoutxlm_line1d import LayoutXLM_line1d_Tokenizer
from .tokenization_layouxtlm_line1d_fast import LayoutXLM_line1d_TokenizerFast


AutoConfig.register("layoutxlm_line1d", LayoutXLM_line1d_Config)
AutoModel.register(LayoutXLM_line1d_Config, LayoutLMCasedModel)
AutoModelForTokenClassification.register(LayoutXLM_line1d_Config, LayoutXLM_line1d_ForTokenClassification)
AutoModelForSequenceClassification.register(LayoutXLM_line1d_Config, LayoutXLM_line1d_ForSequenceClassification)
AutoTokenizer.register(
    LayoutXLM_line1d_Config, slow_tokenizer_class=LayoutXLM_line1d_Tokenizer, fast_tokenizer_class=LayoutXLM_line1d_TokenizerFast
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutXLM_line1d_Tokenizer": XLMRobertaConverter})