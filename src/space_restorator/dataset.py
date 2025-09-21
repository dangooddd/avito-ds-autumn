from datasets import load_dataset


def label_spaces(tokens, tokenizer):
    labels = []
    clean_tokens = []
    for i, tok in enumerate(tokens):
        if tok in tokenizer.all_special_tokens:
            labels.append(-100)
        elif tok.startswith("##"):
            labels.append(0)
        else:
            labels.append(1)
            if i != 1:
                tok = "##" + tok
        clean_tokens.append(tok)

    return clean_tokens, labels


def tokenize_label_spaces(examples, tokenizer):
    encoded = tokenizer(
        examples["text"],
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded["input_ids"]]
    clean_tokens, labels = zip(
        *[label_spaces(token_list, tokenizer) for token_list in tokens]
    )
    clean_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in clean_tokens]

    return {
        "input_ids": clean_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "tokens": tokens,
        "clean_tokens": clean_tokens,
    }


def create_dataset(tokenizer):
    raw_dataset = load_dataset(
        "IlyaGusev/ru_news", split="train", streaming=True, trust_remote_code=True
    )

    tokenized_dataset = raw_dataset.map(
        lambda ex: tokenize_label_spaces(ex, tokenizer),
        batched=True,
        remove_columns=["text", "title", "url", "datetime", "source", "timestamp"],
    )

    return tokenized_dataset
