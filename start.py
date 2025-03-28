import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from main import load_data, tokenize_dataset, create_data_collator, create_trainer
from evaluate import load

ds_path = "hf://datasets/pszemraj/scientific_lay_summarisation-plos-norm/" + "train.parquet"
df = pd.read_parquet(ds_path)[['article', 'summary']]

datasets = load_data(df[:100])

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
rouge = load("rouge")

data_collator = create_data_collator(tokenizer, model)
trainer = create_trainer(model, tokenized_datasets, data_collator, tokenizer)

trainer.train()

model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")