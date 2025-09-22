from argparse import ArgumentParser
from .cascade import cascade, MODEL_GAP, MODEL_SPACE
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path
import pandas as pd


def find_pos(text: str, pred: str):
    pos = []
    o = 0
    for i, ch in enumerate(text):
        if pred[i + o] == " ":
            o += 1
            pos.append(i)

    return pos


def string_from_pos(pos: list):
    result = ", ".join([str(i) for i in pos])
    return f"[{result}]"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--pretrained-space", type=str, default=MODEL_SPACE)
    parser.add_argument("--pretrained-gap", type=str, default=MODEL_GAP)
    parser.add_argument("--spaces", type=float, default=0)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--min-tries", type=int, default=1)
    parser.add_argument("--save-path", type=Path, required=True)
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    tokenizer_gap = AutoTokenizer.from_pretrained(args.pretrained_gap)
    tokenizer_space = AutoTokenizer.from_pretrained(args.pretrained_space)
    model_gap = AutoModelForTokenClassification.from_pretrained(args.pretrained_gap).to(
        device
    )
    model_space = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_space
    ).to(device)

    df = pd.read_csv(args.file)
    predictions = [
        cascade(
            model_gap=model_gap,
            model_space=model_space,
            tokenizer_gap=tokenizer_gap,
            tokenizer_space=tokenizer_space,
            max_tries=args.max_tries,
            min_tries=args.max_tries,
            spaces=args.spaces,
            text=text,
        )
        for text in df["text_no_spaces"]
    ]
    positions = [
        find_pos(text, pred) for text, pred in zip(df["text_no_spaces"], predictions)
    ]
    results = [string_from_pos(pos) for pos in positions]
    df = pd.DataFrame({"id": df["id"], "predicted_positions": results})

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_path, index=False)
