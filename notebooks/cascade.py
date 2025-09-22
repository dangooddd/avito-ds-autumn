# %%
from space_restorator.bert.predict import predict as space_predict
from space_restorator.bert_gap.predict import predict as gap_predict
from space_restorator.bert_gap.dataset import insert_random_spaces_with_indices
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re


def cascade(
    model_space, model_gap, tokenizer_space, tokenizer_gap, text, tries, spaces
):
    for i in range(tries):
        text, _ = insert_random_spaces_with_indices(text, int(len(text) * 0.5))
        text = gap_predict([text], tokenizer_gap, model_gap)[0]
        text = space_predict([text], tokenizer_space, model_space)[0]
        text = re.sub(r"\s+", " ", text)

    text = gap_predict([text], tokenizer_gap, model_gap)[0]
    return text


MODEL_GAP = "../data/output-gap-1/checkpoint-1000"
MODEL_SPACE = "../data/output-8/checkpoint-10000"
tokenizer_gap = AutoTokenizer.from_pretrained(MODEL_GAP)
tokenizer_space = AutoTokenizer.from_pretrained(MODEL_SPACE)
model_gap = AutoModelForTokenClassification.from_pretrained(MODEL_GAP)
model_space = AutoModelForTokenClassification.from_pretrained(MODEL_SPACE)

text = "новыйдиванIkea"
result = cascade(
    model_space=model_space,
    model_gap=model_gap,
    tokenizer_space=tokenizer_space,
    tokenizer_gap=tokenizer_gap,
    text=text,
    tries=5,
    spaces=0.4,
)

print(result)
