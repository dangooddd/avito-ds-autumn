from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from . import MODEL_NAME
from .dataset import create_dataset

from argparse import ArgumentParser


def train(
    lr=3e-5,
    steps=1000,
    batch_size=16,
    output_dir="data/output",
    pretrained=MODEL_NAME,
    acc=1,
    resume=False,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained,
        num_labels=2,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    dataset = create_dataset(tokenizer)
    eval_dataset_size = 2500
    eval_dataset = dataset.take(eval_dataset_size)
    train_dataset = dataset.skip(eval_dataset_size)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir="data/logs",
        eval_strategy="steps",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=3,
        eval_steps=500,
        logging_strategy="steps",
        gradient_accumulation_steps=acc,
        lr_scheduler_type="cosine_with_restarts",
        logging_steps=10,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=steps,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.evaluate()


def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = np.argmax(pred.predictions, axis=-1).flatten()
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--acc", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--pretrained", type=str, default=f"{MODEL_NAME}")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        resume=args.resume,
        acc=args.acc,
    )
