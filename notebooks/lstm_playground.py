import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random


class SpaceDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        for text in texts:
            clean = text.replace(" ", "")
            labels = [0] * len(clean)

            j = 0
            for i, ch in enumerate(clean):
                if j < len(text) and text[j] == " ":
                    labels[i] = 1
                    j += 1
                j += 1
            self.samples.append((clean, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, labels = self.samples[idx]
        ids = torch.tensor(self.tokenizer(text), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return ids, labels


class CharTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1}

    def build_vocab(self, texts):
        for text in texts:
            for ch in text:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text):
        return [self.vocab.get(ch, 1) for ch in text]

    def decode(self, ids):
        return "".join([self.inv_vocab[i] for i in ids])


class SpaceRestorer(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 0 = нет пробела, 1 = есть пробел

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits


def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 2), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss={total_loss / len(dataloader):.4f}")


def restore_spaces(model, tokenizer, text):
    model.eval()
    ids = torch.tensor([tokenizer(text)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        preds = logits.argmax(-1).squeeze(0).tolist()
    result = []
    for ch, p in zip(text, preds):
        if p == 1:
            result.append(" ")
        result.append(ch)
    return "".join(result)


train_texts = [
    "купить айфон 14 про",
    "заказать макбук про",
    "новый самсунг галакси",
    "посмотреть фильм дома",
]

tokenizer = CharTokenizer()
tokenizer.build_vocab(train_texts)

dataset = SpaceDataset(train_texts, tokenizer)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda batch: (
        nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
        nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True),
    ),
)

model = SpaceRestorer(len(tokenizer.vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # игнорируем паддинги

train_model(model, dataloader, optimizer, criterion, epochs=15)

test_text = "купитьайфон14про"
print("Input:", test_text)
print("Restored:", restore_spaces(model, tokenizer, test_text))
