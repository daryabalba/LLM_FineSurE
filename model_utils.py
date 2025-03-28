import numpy as np
import torch
from evaluate import load
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                         DataCollatorForSeq2Seq, Seq2SeqTrainer,
                         Seq2SeqTrainingArguments)

from config import MODEL_DIR, MODEL_NAME

rouge = load("rouge")


def load_data(df):
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    return {
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    }


def tokenize_sample(sample, tokenizer, max_length, max_target_length):
    source_encodings = tokenizer(
        sample['article'],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    target_encodings = tokenizer(
        sample['summary'],
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt"
    )

    return {
        "input_ids": source_encodings["input_ids"].squeeze(),
        "attention_mask": source_encodings["attention_mask"].squeeze(),
        "labels": target_encodings["input_ids"].squeeze()
    }


def tokenize_dataset(dataset, tokenizer, max_length, max_target_length):
    return dataset.map(
        lambda x: tokenize_sample(x, tokenizer, max_length, max_target_length),
        batched=False,
        remove_columns=['article', 'summary']
    )


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )

    return {k: round(v, 4) for k, v in result.items()}


def create_data_collator(tokenizer, model, padding=True):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                  model=model,
                                  padding=padding)


def create_trainer(model, dataset, data_collator, tokenizer, num_train_epochs=5):
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR),
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        gradient_accumulation_steps=2,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    return trainer


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer


def generate_summary(text, model, tokenizer, max_input_length=1024, max_new_tokens=100):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
