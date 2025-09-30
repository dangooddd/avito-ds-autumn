from .bert.predict import predict as space_predict
from .bert_gap.predict import predict as gap_predict
from .bert_gap.dataset import insert_random_spaces_with_indices
from transformers import AutoTokenizer, AutoModelForTokenClassification
from argparse import ArgumentParser
import re
import torch


MODEL_GAP = "models/checkpoint-gap"
MODEL_SPACE = "models/checkpoint-space"

# Словарь для помощи модели в детекции самостоятельных слов, плохо ею изученных.
# Данные слова самостоятельны и не встречаются как подстроки других слов.
dictionary = {
    "новый",
    "куплю",
    "новая",
    "айфон",
    "iphone",
    "продам",
    "ремонт",
}


def split_string_with_spaces(text: str):
    """
    Разделяет все буквы текста на отдельные слова.
    Пример: "Приветмир" -> "П р и в е т м и р"

    Args:
        text: входная строка

    Returns:
        output: строка, в которой все буквы разделены на отдельные слова.
    """
    text = " ".join(list(text))
    return re.sub(r"\s+", " ", text)


def add_spaces_around_words(text: str, words: set):
    """
    Выделяет слова из словаря пробелами.

    Args:
        text: исходный текст
        words: словарь слов, которые необходимо выделить

    Returns:
        output: str
    """
    escaped_words = [re.escape(w) for w in words]
    pattern = r"(" + "|".join(escaped_words) + r")"

    def replacer(match):
        word = match.group(0)
        start, end = match.start(), match.end()

        left_ok = (start == 0) or (text[start - 1] == " ")
        right_ok = (end == len(text)) or (text[end] == " ")

        left_space = "" if left_ok else " "
        right_space = "" if right_ok else " "

        return f"{left_space}{word}{right_space}"

    result = re.sub(pattern, replacer, text)
    return re.sub(r"\s+", " ", result)


def cascade(
    model_space,
    model_gap,
    tokenizer_space,
    tokenizer_gap,
    text,
    max_tries,
    min_tries,
    spaces,
):
    global dictionary
    text = split_string_with_spaces(text)
    prev = None
    for i in range(max_tries):
        text = gap_predict([text], tokenizer_gap, model_gap)[0]
        text = add_spaces_around_words(text, dictionary)
        text = space_predict([text], tokenizer_space, model_space)[0]
        if spaces > 0 and i < min_tries:
            text, _ = insert_random_spaces_with_indices(text, int(len(text) * spaces))
        text = re.sub(r"\s+", " ", text)

        if prev == text and i >= min_tries:
            break

        prev = text

    text = gap_predict([text], tokenizer_gap, model_gap)[0].strip()
    text = add_spaces_around_words(text, dictionary)
    return text


def load_models(
    model_space_pretrained: str = MODEL_SPACE,
    model_gap_pretrained: str = MODEL_GAP,
    device: str = "cpu",
):
    """
    Инициализирует space и gap модели.

    Args:
        model_space_pretrained: путь к чекпоинту space модели
        model_gap_pretrained: путь к чекпоинту gap модели
        device: device для инициализации

    Returns:
        model_space: space модель
        model_gap: gap модель
        tokenizer_space: токенизатор space модели
        tokenizer_gap: токенизатор gap модели
    """

    tokenizer_gap = AutoTokenizer.from_pretrained(model_gap_pretrained)
    tokenizer_space = AutoTokenizer.from_pretrained(model_space_pretrained)
    model_gap = AutoModelForTokenClassification.from_pretrained(
        model_gap_pretrained
    ).to(device)
    model_space = AutoModelForTokenClassification.from_pretrained(
        model_space_pretrained
    ).to(device)

    return model_space, model_gap, tokenizer_space, tokenizer_gap


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained-space",
        type=str,
        default=MODEL_SPACE,
        help="Модель вставки пробелов. Может быть как названием модели, так и путем к чекпоинту.",
    )
    parser.add_argument(
        "--pretrained-gap",
        type=str,
        default=MODEL_GAP,
        help="Модель склейки. Может быть как названием модели, так и путем к чекпоинту.",
    )
    parser.add_argument(
        "--spaces",
        type=float,
        default=0.2,
        help="Процент пробелов, вставляемый на каждой итерации каскадного алгоритма. Если 0, действует детерминированный режим.",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=15,
        help="Максимальное число итераций каскадного алгоритма.",
    )
    parser.add_argument(
        "--min-tries",
        type=int,
        default=7,
        help="Минимальное число итераций каскадного алгоритма.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Принудительно использовать CPU",
    )
    parser.add_argument("text", type=str, help="Входная строка")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Используется устройство: {device}")

    model_space, model_gap, tokenizer_space, tokenizer_gap = load_models(
        model_space_pretrained=args.pretrained_space,
        model_gap_pretrained=args.pretrained_gap,
        device=device,
    )

    result = cascade(
        model_space=model_space,
        model_gap=model_gap,
        tokenizer_space=tokenizer_space,
        tokenizer_gap=tokenizer_gap,
        text=args.text,
        max_tries=args.max_tries,
        min_tries=args.max_tries,
        spaces=args.spaces,
    )

    print(result)
