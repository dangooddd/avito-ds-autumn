# %%
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
from torch.utils.data import Dataset

MODEL_NAME = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)


class SpaceDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            padding=False,
        )
        input_ids = encoding["input_ids"]
        # Добавляем метки для [CLS] и [SEP]
        padded_labels = [0] + label + [0]
        padded_labels = padded_labels[: len(input_ids)]
        encoding["labels"] = padded_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


# Пример данных
texts = ["куплюайфон14про"]
labels = [[0, 1, 0, 0, 1, 0, 0]]
dataset = SpaceDataset(texts, labels)

# Data collator с паддингом
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./ruberta-space",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset, data_collator=data_collator
)

trainer.train()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Переносим модель на устройство и в режим оценки
model.to(device)
model.eval()


def restore_spaces(model, tokenizer, text: str, device=device):
    # Токенизация с возвратом input_ids и attention_mask
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    # Переносим тензоры на выбранное устройство
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Получаем логиты модели
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape (1, seq_len, num_labels)

    # Прогноз меток (0 или 1) для каждого токена
    preds = torch.argmax(logits, dim=-1).squeeze().tolist()

    # Получаем токены (включая [CLS], [SEP])
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    # Собираем итоговую строку с пробелами
    result = []
    for token, label in zip(tokens, preds):
        if token in tokenizer.all_special_tokens:
            continue  # пропускаем специальные токены
        piece = token.replace("##", "")
        result.append(piece)
        if label == 1:
            result.append(" ")

    return "".join(result)


# Пример использования
text = "куплюайфон14про"
restored = restore_spaces(model, tokenizer, text)
print(restored)  # Ожидаемый вывод: "куплю айфон14 про"
