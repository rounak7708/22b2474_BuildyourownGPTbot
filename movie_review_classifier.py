!pip uninstall -y transformers
!pip install -q transformers==4.41.1

# ✅ Install only once
!pip install -q transformers datasets

# ✅ Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()
for p in bert.parameters():
    p.requires_grad = False

# ✅ Dataset Class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_mask[idx], self.labels[idx]

# ✅ Small subset for quick demo
train_data = load_dataset("imdb", split="train[:10%]")  # ~250 samples
train_ds = IMDBDataset(train_data["text"], train_data["label"])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

# ✅ Model using frozen BERT + LSTM
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(768, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            h = bert(input_ids, attention_mask=attention_mask).last_hidden_state
        _, (h_n, _) = self.lstm(h)
        return self.fc(h_n[-1])

model = LSTMClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ✅ Train for 3 epochs
for epoch in range(3):
    model.train()
    total_loss = 0
    for x, m, y in train_dl:
        x, m, y = x.to(device), m.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, m)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_dl):.4f}")

def predict(text):
    model.eval()
    tok = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(tok['input_ids'], tok['attention_mask'])
        probs = torch.softmax(out, dim=1)
        label = torch.argmax(probs, dim=1).item()
        return "Positive" if label == 1 else "Negative"

print("👉", predict("I loved the movie!"))      # ✅ Expected: Positive
print("👉", predict("It was horrible."))        # ✅ Expected: Negative


def evaluate():
    test_data = load_dataset("imdb", split="test[:10%]")  # Smaller subset
    test_ds = IMDBDataset(test_data["text"], test_data["label"])
    test_dl = DataLoader(test_ds, batch_size=16)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, m, y in test_dl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            out = model(x, m)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Test Accuracy: {acc:.4f}")

evaluate()
