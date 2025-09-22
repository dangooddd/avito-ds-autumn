from .bert.predict import predict as space_predict
from .bert_gap.predict import predict as gap_predict
from .bert_gap.dataset import insert_random_spaces_with_indices
from transformers import AutoTokenizer, AutoModelForTokenClassification
from argparse import ArgumentParser
import re


MODEL_GAP = "models/checkpoint-gap"
MODEL_SPACE = "models/checkpoint-space"

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
    text = " ".join(list(text))
    return re.sub(r"\s+", " ", text)


def add_spaces_around_words(text, words):
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
    text = split_string_with_spaces(text)
    prev = None
    for i in range(max_tries):
        text = gap_predict([text], tokenizer_gap, model_gap)[0]
        text = add_spaces_around_words(text, dictionary)
        if spaces > 0:
            text = space_predict([text], tokenizer_space, model_space)[0]
        text, _ = insert_random_spaces_with_indices(text, int(len(text) * spaces))
        text = re.sub(r"\s+", " ", text)
        if prev == text and i >= min_tries:
            break

        prev = text

    text = gap_predict([text], tokenizer_gap, model_gap)[0].strip()
    text = add_spaces_around_words(text, dictionary)
    print(text)
    return text


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained-space", type=str, default=MODEL_SPACE)
    parser.add_argument("--pretrained-gap", type=str, default=MODEL_GAP)
    parser.add_argument("--spaces", type=float, default=0.4)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--min-tries", type=int, default=3)
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    tokenizer_gap = AutoTokenizer.from_pretrained(args.pretrained_gap)
    tokenizer_space = AutoTokenizer.from_pretrained(args.pretrained_space)
    model_gap = AutoModelForTokenClassification.from_pretrained(args.pretrained_gap)
    model_space = AutoModelForTokenClassification.from_pretrained(args.pretrained_space)

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
