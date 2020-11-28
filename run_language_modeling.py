"""
@Time   :   2020-11-26 17:03:29
@File   :   run_pretraining.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""

from sklearn.model_selection import train_test_split
import os
from simpletransformers.language_modeling import LanguageModelingModel
import logging


def proc_data():
    import json
    all_text = []
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
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "block_size": 128,
        "max_seq_length": 512,
        "dataset_type": "simple",
        "wandb_project": "Esperanto - ConvBert",
        "wandb_kwargs": {"name": "ConvBert-SMALL"},
        "logging_steps": 100,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 300,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": True,
        "sliding_window": True,
        "vocab_size": 52000,
        "generator_config": {
            "embedding_size": 128,
            "hidden_size": 256,
            "num_hidden_layers": 3,
        },
        "discriminator_config": {
            "embedding_size": 128,
            "hidden_size": 256,
        },
    }

    train_file = "data/train.txt"
    test_file = "data/test.txt"

    model = LanguageModelingModel(
        "convbert",
        None,
        args=train_args,
        train_files=train_file,
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
    # proc_data()
    main()
    # save_best_model()
