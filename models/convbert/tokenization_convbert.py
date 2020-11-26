"""
@Time   :   2020-11-26 10:23:47
@File   :   tokenization_convbert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from transformers.tokenization_bert import BertTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/electra-small-generator": "https://huggingface.co/google/electra-small-generator/resolve/main/vocab.txt",
        "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/vocab.txt",
        "google/electra-large-generator": "https://huggingface.co/google/electra-large-generator/resolve/main/vocab.txt",
        "google/electra-small-discriminator": "https://huggingface.co/google/electra-small-discriminator/resolve/main/vocab.txt",
        "google/electra-base-discriminator": "https://huggingface.co/google/electra-base-discriminator/resolve/main/vocab.txt",
        "google/electra-large-discriminator": "https://huggingface.co/google/electra-large-discriminator/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}


class ConvBertTokenizer(BertTokenizer):
    r"""
    Construct an ConvBERT tokenizer.
    :class:`~transformers.ConvBertTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.
    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
