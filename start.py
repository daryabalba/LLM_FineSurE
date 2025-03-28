import pandas as pd
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from main import (create_data_collator, create_trainer, load_data,
                  tokenize_dataset)


def main():
    ds_path = "hf://datasets/pszemraj/scientific_lay_summarisation-plos-norm/" + "train.parquet"
    df = pd.read_parquet(ds_path)[['article', 'summary']]

    datasets = load_data(df[:10])

    MAX_LENGTH = 1000
    MAX_TARGET_LENGTH = 200
    MODEL_NAME = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_datasets = {
        'train': tokenize_dataset(datasets['train'], tokenizer,
                                  max_length=MAX_LENGTH,
                                  max_target_length=MAX_TARGET_LENGTH),
        'validation': tokenize_dataset(datasets['validation'], tokenizer,
                                       max_length=MAX_LENGTH,
                                       max_target_length=MAX_TARGET_LENGTH)
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    data_collator = create_data_collator(tokenizer, model)
    trainer = create_trainer(model, tokenized_datasets, data_collator, tokenizer)

    trainer.train()

    model_path = Path(__file__).parent / 'dist' / 'fine_tuned_bart'
    Path(model_path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print("Successfully trained a model")


if __name__ == "__main__":
    main()