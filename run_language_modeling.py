"""
@Time   :   2020-11-26 17:03:29
@File   :   run_pretraining.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""

from sklearn.model_selection import train_test_split
import os

from tqdm import tqdm

from simpletransformers.language_modeling import LanguageModelingModel
import logging


def proc_data():
    import json
    all_text = []
    data_path = '/ml/nlp/data'
    # for fn in os.listdir(data_path):
    #     if os.path.isdir(os.path.join(data_path, fn)):
    #         for txt_name in tqdm(os.listdir(os.path.join(data_path, fn))):
    #             txt_path = os.path.join(data_path, fn, txt_name)
    #             if txt_path.endswith('.txt'):
    #                 with open(txt_path, 'r', encoding='utf8') as f:
    #                     for line in f:
    #                         line = line.strip()
    #                         if len(line) > 1:
    #                             all_text.append(line)
    data_path = '/ml/nlp/data/wiki'
    for txt_name in tqdm(os.listdir(data_path)):
        txt_path = os.path.join(data_path, txt_name)
        if txt_path.endswith('.txt'):
            with open(txt_path, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 1:
                        all_text.append(line)

    with open('data/data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    for d in data:
        all_text.append(d['Title'])
        all_text.append(d['Content'])

    train, test = train_test_split(all_text, test_size=0.1)

    with open("data/train.txt", "w") as f:
        for line in train:
            f.write(line + "\n")

    with open("data/test.txt", "w") as f:
        for line in test:
            f.write(line + "\n")


def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_args = {
        "reprocess_input_data": False,
        "overwrite_output_dir": True,
        "num_train_epochs": 50,
        "save_eval_checkpoints": True,
        "save_model_every_epoch": False,
        "learning_rate": 1e-3,
        "warmup_steps": 10000,
        "train_batch_size": 64,
        "eval_batch_size": 128,
        "gradient_accumulation_steps": 2,
        "block_size": 128,
        "max_seq_length": 128,
        "dataset_type": "simple",
        "wandb_project": "Esperanto - ConvBert",
        "wandb_kwargs": {"name": "ConvBert-SMALL"},
        "logging_steps": 100,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 3000,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": True,
        "sliding_window": False,
        "tokenizer_name": "bert-base-chinese",
        "use_multiprocessing": True,
        "process_count": 8,
        "vocab_size": 21128,
        "generator_config": {
            "attention_probs_dropout_prob": 0.1,
            "directionality": "bidi",
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "convbert",
            "num_attention_heads": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "summary_activation": "gelu",
            "summary_last_dropout": 0.1,
            "summary_type": "first",
            "summary_use_proj": True,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
        "discriminator_config": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "convbert",
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "output_past": True,
            "pad_token_id": 0,
            "summary_activation": "gelu",
            "summary_last_dropout": 0.1,
            "summary_type": "first",
            "summary_use_proj": True,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
    }

    train_file = "data/train.txt"
    test_file = "data/test.txt"

    model = LanguageModelingModel(
        "convbert",
        None,
        args=train_args,
        train_files=train_file,
        cuda_device=1,
    )

    model.train_model(
        train_file, eval_file=test_file,
    )

    model.eval_model(test_file)


def save_best_model():
    model = LanguageModelingModel(
        'convbert',
        'outputs/best_model',
        args={"output_dir": "discriminator_trained"}
    )
    model.save_discriminator()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # proc_data()
    main()
    # save_best_model()
