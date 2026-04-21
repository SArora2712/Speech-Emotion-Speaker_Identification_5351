"""
STABLE CNN + GRU EMOTION CLASSIFIER (UPDATED)
Fixes overfitting + training instability

Key improvements:
- SpecAugment (data augmentation)
- Smaller CNN (prevents overfitting)
- GRU instead of LSTM (more stable)
- Label smoothing loss
- Strong dropout
- Gradient clipping
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import joblib
import warnings
warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
RAVDESS_PATH = "data/Ravdess"
MODELS_FOLDER = "models"

SAMPLE_RATE = 22050
N_MELS = 128
MAX_TIME = 128

BATCH_SIZE = 32
EPOCHS = 150
LR = 0.0008
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
OUTPUTS_FOLDER = "outputs"
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# =========================
# EMOTION MAP
# =========================
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}


# =========================
# FEATURE EXTRACTION
# =========================
def extract_mel(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    audio, _ = librosa.effects.trim(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        hop_length=512
    )

    mel = librosa.power_to_db(mel)

    # normalize
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # pad/truncate
    if mel.shape[1] < MAX_TIME:
        pad = MAX_TIME - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)))
    else:
        mel = mel[:, :MAX_TIME]

    return mel[np.newaxis, :, :]


# =========================
# SPEC AUGMENTATION
# =========================
def spec_augment(mel):
    mel = mel.copy()

    # time mask
    t = random.randint(0, 20)
    t0 = random.randint(0, mel.shape[2] - t)
    mel[:, :, t0:t0+t] = 0

    # freq mask
    f = random.randint(0, 15)
    f0 = random.randint(0, mel.shape[1] - f)
    mel[:, f0:f0+f, :] = 0

    return mel


# =========================
# LOAD DATASET
# =========================
def load_ravdess():
    data, labels = [], []

    for root, _, files in os.walk(RAVDESS_PATH):
        for f in files:
            if not f.endswith(".wav"):
                continue

            parts = f.split("-")
            if len(parts) < 3:
                continue

            emotion = RAVDESS_EMOTIONS.get(parts[2])
            if not emotion:
                continue

            path = os.path.join(root, f)
            try:
                mel = extract_mel(path)
                data.append(mel)
                labels.append(emotion)
            except:
                continue

    return np.array(data), np.array(labels)


# =========================
# DATASET CLASS
# =========================
class EmotionDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.augment:
            x = spec_augment(x)

        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()


# =========================
# MODEL (CNN + GRU)
# =========================
class EmotionCNNGRU(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        self.gru = nn.GRU(
            input_size=64 * 16,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x)

        b, c, f, t = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, t, c * f)

        out, _ = self.gru(x)
        out = out[:, -1, :]

        return self.fc(out)


# =========================
# LABEL SMOOTHING LOSS
# =========================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# =========================
# TRAIN
# =========================
def train_epoch(model, loader, opt, loss_fn):
    model.train()

    total, correct, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += loss.item()
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / len(loader), correct / total


def evaluate(model, loader, loss_fn):
    model.eval()

    preds, true, loss_sum = [], [], 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = loss_fn(out, y)

            loss_sum += loss.item()

            preds.extend(out.argmax(1).cpu().numpy())
            true.extend(y.cpu().numpy())

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="weighted")

    return loss_sum / len(loader), acc, f1, preds, true

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, "training_curves_cnn.png"))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, le, acc, f1):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.title("Confusion Matrix (%)", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 🔥 ADD ACCURACY + F1 BELOW MATRIX
    plt.figtext(
        0.5, -0.05,
        f"Accuracy: {acc*100:.2f}%   |   F1 Score: {f1*100:.2f}%",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5}
    )

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_FOLDER, "confusion_matrix_cnn_gru.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved → {save_path}")
# =========================
# MAIN TRAINING PIPELINE
# =========================
def main():
    print("Loading dataset...")

    x, y = load_ravdess()

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    train_ds = EmotionDataset(x_train, y_train, augment=True)
    val_ds = EmotionDataset(x_val, y_val)
    test_ds = EmotionDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = EmotionCNNGRU(len(le.classes_)).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = LabelSmoothingLoss(len(le.classes_), 0.1)

    best_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, opt, loss_fn)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train {train_acc:.3f} | Val {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_FOLDER, "best.pt"))

    print("Best Val Acc:", best_acc)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER, "best.pt")))


    _, test_acc, test_f1, preds, true = evaluate(model, test_loader, loss_fn)

    print("\nTEST RESULTS")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    # 🔥 CONFUSION MATRIX
    plot_confusion_matrix(true, preds, le, test_acc, test_f1)
    torch.save(model.state_dict(), os.path.join(MODELS_FOLDER, "cnn+gru.pt"))

if __name__ == "__main__":
    main()