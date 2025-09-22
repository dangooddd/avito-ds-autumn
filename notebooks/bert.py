# %%
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(
    "xlm-roberta-base", add_prefix_space=True
)
text = "куплюайфон14про"

tokens = tokenizer.tokenize(text)
# tokens[0] = "▁куп"
ids = tokenizer.convert_tokens_to_ids(tokens)

print(tokens)
print(ids)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
text = "куплюайфон14про"

encoded = tokenizer(
    text,
    add_special_tokens=True,  # добавить специальные токены [CLS], [SEP]
    padding=False,  # без паддинга
    truncation=True,  # без усечения
    max_length=3,
    return_tensors=None,  # возвращать списки, а не тензоры
)

tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
token_ids = encoded["input_ids"]

print("Текст:", text)
print("Токены:", tokens)
print("Идентификаторы токенов:", token_ids)

decoded = tokenizer.decode(token_ids)
print("Декодированный текст:", decoded)
