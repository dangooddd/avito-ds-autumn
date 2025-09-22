from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from argparse import ArgumentParser


def restore_spaces(tokens, offsets, labels, tokenizer):
    restored = tokenizer.convert_tokens_to_string(tokens)
    print(tokens)
    print(labels)
    o = 0
    for token, (s, e), label in zip(tokens, offsets, labels):
        if token in tokenizer.all_special_tokens:
            continue
        if label == 1:
            restored = f"{restored[: s + o]} {restored[s + o :]}"
            o += 1

    return restored


def predict(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=512,
        return_tensors="pt",
        is_split_into_words=False,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs["offset_mapping"]
    del inputs["offset_mapping"]

    outputs = model(**inputs)["logits"].argmax(dim=-1)
    restored = []

    for tokens, offsets, labels in zip(
        [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]],
        offset_mapping,
        outputs,
    ):
        restored.append(restore_spaces(tokens, offsets, labels, tokenizer))

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
