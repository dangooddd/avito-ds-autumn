from datasets import load_dataset


def create_dataset():
    raw_dataset = load_dataset(
        "IlyaGusev/pikabu", split="train", streaming=True, trust_remote_code=True
    )

    tokenized_dataset = raw_dataset.map(
        lambda ex: {"text": ex["text_markdown"]},
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
