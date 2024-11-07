# from transformers import AutoTokenizer
#
# from LayoutXLM_local1d.models.cased.configuration_layoutxlm_line1d import LayoutXLM_line1d_Config
# from LayoutXLM_local1d.models.cased.tokenization_layoutxlm_line1d import LayoutXLM_line1d_Tokenizer
# from LayoutXLM_local1d.models import LayoutXLM_line1d_TokenizerFast
# from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, XLMRobertaConverter
#
# # AutoConfig.register("layoutxlm_line1d", LayoutXLM_line1d_Config)
# # AutoModel.register(LayoutXLM_line1d_Config, LayoutLMCasedModel)
# # AutoModelForTokenClassification.register(LayoutXLM_line1d_Config, LayoutXLM_line1d_ForTokenClassification)
# AutoTokenizer.register(
#     LayoutXLM_line1d_Config, slow_tokenizer_class=LayoutXLM_line1d_Tokenizer, fast_tokenizer_class=LayoutXLM_line1d_TokenizerFast
# )
# SLOW_TO_FAST_CONVERTERS.update({"LayoutXLM_line1d_Tokenizer": XLMRobertaConverter})


from .models import (
    LayoutXLM_line1d_Config,
    LayoutXLM_line1d_Tokenizer,
    LayoutXLM_line1d_uncased_Config,
    LayoutXLM_line1d_uncased_Tokenizer,
)