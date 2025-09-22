from datasets import load_dataset
import re
import random


def insert_random_spaces_with_indices(text, num_spaces):
    if num_spaces <= 0 or num_spaces >= len(text):
        return text, []

    positions = list(range(1, len(text)))  # возможные позиции для вставки пробелов
    selected_positions = random.sample(positions, min(num_spaces, len(positions)))
    selected_positions.sort()

    result = list(text)
    offset = 0
    space_indices = []

    for pos in selected_positions:
        pos_with_offset = pos + offset

        if ((pos > 0 and text[pos - 1]) == " ") or (text[pos] == " "):
            continue

        result.insert(pos_with_offset, " ")
        space_indices.append(pos_with_offset)
        offset += 1

    return "".join(result), space_indices


def label_gaps(text, tokenizer):
    new_text = re.sub(r"\s+", " ", text)
    new_text, marks = insert_random_spaces_with_indices(text, int(len(text) * 0.3))
    encoded = tokenizer(
        new_text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    labels = []

    i = 0
    for (s, e), ids in zip(encoded["offset_mapping"], encoded["input_ids"]):
        if ids in tokenizer.all_special_ids:
            labels.append(-100)
        elif i < len(marks) and s - 1 == marks[i]:
            labels.append(1)
            i += 1
        else:
            labels.append(0)

    return (
        encoded["input_ids"],
        tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
        encoded["attention_mask"],
        labels,
    )


def tokenize_label_spaces(examples, tokenizer):
    input_ids, tokens, attention_mask, labels = zip(
        *[label_gaps(text, tokenizer) for text in examples["text_markdown"]]
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tokens": tokens,
    }


def create_dataset(tokenizer):
    # raw_dataset = load_dataset(
    #     "IlyaGusev/ru_news", split="train", streaming=True, trust_remote_code=True
    # )
    raw_dataset = load_dataset(
        "IlyaGusev/pikabu", split="train", streaming=True, trust_remote_code=True
    )

    tokenized_dataset = raw_dataset.map(
        lambda ex: tokenize_label_spaces(ex, tokenizer),
        batched=True,
        remove_columns=[
            "text_markdown",
            "title",
            "url",
            "datetime",
            "source",
            "timestamp",
        ],
    )

    return tokenized_dataset
