# %%
from transformers import AutoTokenizer
from space_restorator.bert_gap.dataset import label_gaps

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = "Альберт Эйнштейн был гением"

label_gaps(text, tokenizer)
