import pandas as pd
from model_utils import (load_data, tokenize_dataset,
                         create_data_collator, create_trainer,
                         load_model_and_tokenizer)
from config import MODEL_NAME, MAX_LENGTH, MAX_TARGET_LENGTH, MODEL_DIR


def train_model():
    ds_path = "hf://datasets/pszemraj/scientific_lay_summarisation-plos-norm/" + "train.parquet"
    df = pd.read_parquet(ds_path)[['article', 'summary']]

    datasets = load_data(df[:10])  # Для демо берем только 10 примеров

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    tokenized_datasets = {
        'train': tokenize_dataset(datasets['train'], tokenizer,
                                  MAX_LENGTH, MAX_TARGET_LENGTH),
        'validation': tokenize_dataset(datasets['validation'], tokenizer,
                                       MAX_LENGTH, MAX_TARGET_LENGTH)
    }

    data_collator = create_data_collator(tokenizer, model)
    trainer = create_trainer(model, tokenized_datasets, data_collator, tokenizer)

    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_model()