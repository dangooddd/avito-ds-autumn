from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
import torch

from .dataset import tokenize_label_spaces
from argparse import ArgumentParser


def restore_spaces(tokens, labels, tokenizer):
    restored_tokens = []
    print(tokens)
    print(labels)
    for token, label in zip(tokens, labels):
        if token in tokenizer.all_special_tokens:
            continue
        if label == 1:
            token = token[2:] if token.startswith("##") else token

        restored_tokens.append(token)

    return tokenizer.convert_tokens_to_string(restored_tokens)


def predict(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        is_split_into_words=False,
    )

    ex = {"text": texts}
    res = tokenize_label_spaces(ex, tokenizer)
    inputs["input_ids"] = torch.tensor(res["input_ids"])

    outputs = model(**inputs)["logits"].argmax(dim=-1)
    restored = []

    for tokens, labels in zip(
        [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]],
        outputs,
    ):
        restored.append(restore_spaces(tokens, labels, tokenizer))

    return restored


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    model = AutoModelForTokenClassification.from_pretrained(args.pretrained)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    texts = [args.text]
    print(predict(texts=texts, model=model, tokenizer=tokenizer))
