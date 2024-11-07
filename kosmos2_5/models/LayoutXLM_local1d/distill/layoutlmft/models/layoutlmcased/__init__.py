from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from .configuration_layoutlmcased import LayoutLMCasedConfig
from .modeling_layoutlmcased import (
    LayoutLMCasedForCausalLM,
    LayoutLMCasedForMaskedLM,
    LayoutLMCasedForTokenClassification,
    LayoutLMCasedModel,
)
from .tokenization_layoutlmcased import LayoutLMCasedTokenizer
from .tokenization_layoutlmcased_fast import LayoutLMCasedTokenizerFast


AutoConfig.register("layoutlmcased", LayoutLMCasedConfig)
AutoModel.register(LayoutLMCasedConfig, LayoutLMCasedModel)
AutoModelForMaskedLM.register(LayoutLMCasedConfig, LayoutLMCasedModel)
AutoModelForTokenClassification.register(LayoutLMCasedConfig, LayoutLMCasedForTokenClassification)
AutoTokenizer.register(
    LayoutLMCasedConfig, slow_tokenizer_class=LayoutLMCasedTokenizer, fast_tokenizer_class=LayoutLMCasedTokenizerFast
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMCasedTokenizer": RobertaConverter})
