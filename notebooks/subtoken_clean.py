# %%
from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    DatasetDict,
    IterableDataset,
    Features,
    Sequence,
    Value,
)

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
text = "куплю айфон 14про"

encoded = tokenizer(text, add_special_tokens=True)
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
leading_spaces = []
clean_tokens = []
for i, tok in enumerate(tokens):
    if tok.startswith("##"):
        leading_spaces.append(0)
        clean_tokens.append(tok[2:])
    else:
        leading_spaces.append(1)
        clean_tokens.append(tok)

print(tokens)
print(clean_tokens)
print(leading_spaces)
print(tokenizer.convert_tokens_to_ids(clean_tokens))


# %%

raw_datasets = load_dataset(
    "IlyaGusev/ru_news", split="train", streaming=True, trust_remote_code=True
)


def label_spaces(tokens, tokenizer):
    labels = []
    clean_tokens = []
    for i, tok in enumerate(tokens):
        if tok in tokenizer.all_special_tokens:
            labels.append(-100)
            clean_tokens.append(tok)
        elif tok.startswith("##"):
            labels.append(0)
            clean_tokens.append(tok[2:])
        else:
            labels.append(1)
            clean_tokens.append(tok)
    return clean_tokens, labels


def tokenize_label_spaces(examples):
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
    }


tokenized_dataset = raw_datasets.map(
    tokenize_label_spaces,
    batched=True,
    remove_columns=["text", "title", "url", "datetime", "source", "timestamp"],
)

print(tokenized_dataset)
# print(next(tokenized_dataset.iter(2)))
print(list(tokenized_dataset.take(2)))
