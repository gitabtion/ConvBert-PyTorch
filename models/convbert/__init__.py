"""
@Time   :   2020-11-19 12:13:52
@File   :   __init__.py.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from transformers.file_utils import is_tokenizers_available, is_torch_available
from .configuration_convbert import CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig
from .tokenization_convbert import ConvBertTokenizer


if is_tokenizers_available():
    from .tokenization_convbert_fast import ConvBertTokenizerFast

if is_torch_available():
    from .modeling_convbert import (
        CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        ConvBertForMaskedLM,
        ConvBertForMultipleChoice,
        ConvBertForPreTraining,
        ConvBertForQuestionAnswering,
        ConvBertForSequenceClassification,
        ConvBertForTokenClassification,
        ConvBertModel,
        ConvBertPreTrainedModel,
        load_tf_weights_in_convbert,
    )