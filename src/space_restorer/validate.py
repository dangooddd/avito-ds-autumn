from datasets import load_dataset
from .cascade import load_models, cascade, MODEL_GAP, MODEL_SPACE
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import re
import torch


def find_pos(text: str, pred: str):
    """Находит позицию вставленных пробелов согласно условию задачи"""
    pos = []
    o = 0
    for i, ch in enumerate(text):
        if (i + o) < len(pred) and pred[i + o] == " ":
            o += 1
            pos.append(i)

    return pos


def f1_from_pos(pos_pred: list[int], pos_label: list[int]) -> float:
    pred_set = set(pos_pred)
    label_set = set(pos_label)
    TP = len(pred_set.intersection(label_set))
    FP = len(pred_set - label_set)
    FN = len(label_set - pred_set)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1_score


def create_dataset():
    dataset = load_dataset(
        "IlyaGusev/ru_news", split="train", streaming=True, trust_remote_code=True
    )

    return dataset


def validate_text(
    models: tuple[object],
    text: str,
    spaces: float,
    max_tries: int,
    min_tries: int,
):
    text = re.sub(r"\s+", " ", text)
    text_clean = re.sub(r"\s+", "", text)
    text_pred = cascade(
        *models,
        text=text_clean,
        spaces=spaces,
        max_tries=max_tries,
        min_tries=min_tries,
    )
    pos_label = find_pos(text_clean, text)
    pos_pred = find_pos(text_clean, text_pred)
    return f1_from_pos(pos_pred=pos_pred, pos_label=pos_label)


def validate(
    models: tuple[object],
    dataset: list[str],
    spaces: float,
    max_tries: int,
    min_tries: int,
):
    f1_list = []
    for text in tqdm(dataset, desc="Валидация"):
        text_f1 = validate_text(
            models,
            text=text,
            spaces=spaces,
            max_tries=max_tries,
            min_tries=min_tries,
        )
        f1_list.append(text_f1)

    f1_avg = pd.Series(f1_list).mean()
    return f1_avg


def main(
    samples: int,
    save_path: Path,
    pretrained_gap: str,
    pretrained_space: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    dataset = []
    for ex in create_dataset().take(samples):
        text = ex["text"]
        dataset.extend(filter(lambda t: len(t.strip()) != 0, re.split(r"[.\n]+", text)))

    grid = {
        "spaces": [0.0, 0.1, 0.2, 0.4],
        "max_tries": [1, 3, 7, 15],
        "min_tries": [0, 1, 3, 7],
    }

    models = load_models(
        model_space_pretrained=pretrained_space,
        model_gap_pretrained=pretrained_gap,
        device=device,
    )

    results = []

    for spaces in grid["spaces"]:
        for max_tries in grid["max_tries"]:
            for min_tries in grid["min_tries"]:
                if min_tries > max_tries:
                    continue
                if spaces > 0 and min_tries == 0:
                    continue

                print(
                    f"Валидация для параметров: spaces = {spaces}, max_tries = {max_tries}, min_tries = {min_tries}"
                )

                f1 = validate(
                    models,
                    dataset=dataset,
                    spaces=spaces,
                    min_tries=min_tries,
                    max_tries=max_tries,
                )

                results.append(
                    {
                        "f1": f1,
                        "spaces": spaces,
                        "max_tries": max_tries,
                        "min_tries": min_tries,
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)


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
        "--samples",
        type=int,
        default=100,
        help="Число документов, используемых для валидации.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="data/output/val.csv",
        help="Пусть для сохранения выходного файла.",
    )
    args = parser.parse_args()

    main(
        samples=args.samples,
        save_path=args.save_path,
        pretrained_gap=args.pretrained_gap,
        pretrained_space=args.pretrained_space,
    )
