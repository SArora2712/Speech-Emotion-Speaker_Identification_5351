# src/train_wav2vec_90_target.py
import os
import torch
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import warnings
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
warnings.filterwarnings("ignore")

# ================= CONFIG (Stable & Optimized) =================
CSV_PATH = "data/wav2vec_ready_dataset.csv"
MODEL_OUTPUT = "models/wav2vec2_final"
BATCH_SIZE = 6                  # Safe batch size
EPOCHS = 40
LR = 5e-6                       # Low learning rate for stability
TARGET_SR = 16000
MAX_LENGTH = TARGET_SR * 7      # 7 seconds
PATIENCE = 6
WEIGHT_DECAY = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ================= DATASET =================
class SERDataset(Dataset):
    def __init__(self, df, processor, label_encoder):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.le = label_encoder
        self.df["label"] = self.le.transform(self.df["emotion"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = sf.read(row["file_path"])
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        if len(waveform) > MAX_LENGTH:
            waveform = waveform[:MAX_LENGTH]
        else:
            waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)))
        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True
        )
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.df.loc[idx, "label"], dtype=torch.long)
        }
def plot_training_curves(train_losses, train_accs, val_accs):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # ---- LOSS ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # ---- ACCURACY ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/wav2vec_final_training.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentage
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # ---- METRICS ----
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # ---- ADD TEXT BELOW ----
    plt.figtext(
        0.5, -0.05,
        f"Accuracy: {acc*100:.2f}%   |   F1 Score: {f1*100:.2f}%",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5}
    )

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_wav2vec_final.png", bbox_inches="tight")
    plt.show()
def main():
    print("Starting Wav2Vec2 Training (Stable Version )...\n")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(df)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df["emotion"])
    joblib.dump(label_encoder, "models/label_encoder_emotion.pkl")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["emotion"], random_state=42
    )

    # Using stable base model (less likely to cause OOM error)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h", 
        num_labels=len(label_encoder.classes_)
    )
    model.to(DEVICE)

    train_dataset = SERDataset(train_df, processor, label_encoder)
    val_dataset = SERDataset(val_df, processor, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_accs = []

    all_preds = []
    all_labels = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad()
            inputs = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        train_acc = correct / total
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(train_acc)
        # Validation
        model.eval()
        correct = total = 0

        epoch_preds = []
        epoch_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_accs.append(val_acc)

        # Save last epoch predictions
        all_preds = epoch_preds
        all_labels = epoch_labels
        
       
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(MODEL_OUTPUT)
            processor.save_pretrained(MODEL_OUTPUT)
            print(f" Best model saved! Val Acc: {best_acc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    print("\nTraining Completed!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    plot_training_curves(train_losses, train_accs, val_accs)
    
    plot_confusion_matrix(all_labels, all_preds, label_encoder)


if __name__ == "__main__":
    main()