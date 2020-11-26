"""
@Time   :   2020-11-19 13:51:03
@File   :   configuration_convbert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/electra-small-generator": "https://huggingface.co/google/electra-small-generator/resolve/main/config.json",
    "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/config.json",
    "google/electra-large-generator": "https://huggingface.co/google/electra-large-generator/resolve/main/config.json",
    "google/electra-small-discriminator": "https://huggingface.co/google/electra-small-discriminator/resolve/main/config.json",
    "google/electra-base-discriminator": "https://huggingface.co/google/electra-base-discriminator/resolve/main/config.json",
    "google/electra-large-discriminator": "https://huggingface.co/google/electra-large-discriminator/resolve/main/config.json",
}


class ConvBertConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=30522,
            embedding_size=128,
            hidden_size=256,
            num_hidden_layers=12,
            num_attention_heads=4,
            intermediate_size=1024,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            summary_type="first",
            summary_use_proj=True,
            summary_activation="gelu",
            summary_last_dropout=0.1,
            pad_token_id=0,
            head_ratio=2,
            linear_groups=1,
            conv_kernel_size=9,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.head_ratio = head_ratio
        self.conv_kernel_size = conv_kernel_size
        self.linear_groups = linear_groups
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
