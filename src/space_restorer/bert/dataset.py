from datasets import load_dataset
import re


def label_spaces(text: str, tokenizer):
    """
    Размечает токены текста без пробелов:
        - 0 если до токена в исходном тексте нет пробела;
        - 1 если есть пробел.

    Args:
        text: размечаемый текст
        tokenizer: токенизатор, используемый для разметки

    Returns:
        input_ids: input_ids полученные из текста без пробелов
        tokens: токены текста без пробелов
        attention_mask: attention mask токенов текста без пробелов
        labels: созданные метки
    """
    text = re.sub(r"\s+", " ", text)
    clean = text.replace(" ", "")
    encoded = tokenizer(
        clean,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    labels = []

    o = 0  # разница между исходным текстом и текстом без пробелов
    for (s, e), ids in zip(encoded["offset_mapping"], encoded["input_ids"]):
        if ids in tokenizer.all_special_ids:
            labels.append(-100)
        elif s > 0 and text[s + o] == " ":
            labels.append(1)
            o += 1
        else:
            labels.append(0)

    return (
        encoded["input_ids"],
        tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
        encoded["attention_mask"],
        labels,
    )


def tokenize_label_spaces(examples, tokenizer):
    """
    Размечает датасет. Подробнее см. `label_spaces`.

    Args:
        examples: один элемент датасета
        tokenizer: токенизатор, используемый для разметки

    Returns:
        input_ids: input_ids полученные из текста датасета
        tokens: токены текста элемента датасета
        attention_mask: attention mask токенов текста элемента датасета
        labels: созданные метки
    """
    input_ids, tokens, attention_mask, labels = zip(
        *[label_spaces(text, tokenizer) for text in examples["text_markdown"]]
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tokens": tokens,
    }


def create_dataset(tokenizer):
    """Создает размеченный датасет"""
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
