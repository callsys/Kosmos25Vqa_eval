from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter, XLMRobertaConverter

from .configuration_layoutxlm import LayoutXLMConfig
from .modeling_layoutxlm import (
#     LayoutLMCasedForCausalLM,
    LayoutXLMForMaskedLM,
#     LayoutLMCasedForMaskedLM,
    LayoutXLMForTokenClassification,
#     LayoutLMCasedModel,
)
from .tokenization_layoutxlm import LayoutXLMTokenizer
from .tokenization_layouxtlm_fast import LayoutXLMTokenizerFast


AutoConfig.register("layoutxlm", LayoutXLMConfig)
# AutoModel.register(LayoutLMCasedConfig, LayoutLMCasedModel)
# AutoModelForMaskedLM.register(LayoutLMCasedConfig, LayoutLMCasedModel)
# AutoModelForTokenClassification.register(LayoutLMCasedConfig, LayoutLMCasedForTokenClassification)
AutoTokenizer.register(
    LayoutXLMConfig, slow_tokenizer_class=LayoutXLMTokenizer, fast_tokenizer_class=LayoutXLMTokenizerFast
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutXLMTokenizer": XLMRobertaConverter})
