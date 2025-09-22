from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from argparse import ArgumentParser
import torch


def remove_gaps(tokens, offsets, labels, tokenizer):
    """
    Производит `склейку` токенизированного текста

    Returns:
        text: строка с меньшим числом пробелов, согласно предсказанию модели labels
    """
    restored = []
    for token, (s, e), label in zip(tokens, offsets, labels):
        if token in tokenizer.all_special_tokens:
            continue
        if label == 1 and token[0] == "▁":
            token = token[1:]

        restored.append(token)

    return tokenizer.convert_tokens_to_string(restored)


def predict(texts, tokenizer, model):
    """
    Убирает пробелы в тексте батчами. См. `remove_gaps`

    Returns:
        results: батч с результатами
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=512,
        return_tensors="pt",
        is_split_into_words=False,
        return_offsets_mapping=True,
    ).to(model.device)

    offset_mapping = inputs["offset_mapping"]
    del inputs["offset_mapping"]

    # Перемещаем входные данные на то же устройство, что и модель
    # inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)["logits"].argmax(dim=-1).cpu()
    restored = []

    for tokens, offsets, labels in zip(
        [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]],
        offset_mapping,
        outputs,
    ):
        restored.append(remove_gaps(tokens, offsets, labels, tokenizer))

    return restored


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    texts = [args.text]
    print(predict(texts=texts, model=model, tokenizer=tokenizer))
