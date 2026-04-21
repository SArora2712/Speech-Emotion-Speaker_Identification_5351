"""
CNN EMOTION MODEL — DAY 1
Mel Spectrogram + 2D CNN on RAVDESS
Target: 85%+ accuracy (vs current 72% MFCC+SVM)

Why CNN beats MFCC+SVM:
  MFCC mean → 40 numbers → loses ALL time information
  CNN mel spectrogram → 128x128 image → sees HOW emotion
  evolves across time + frequency simultaneously

Run: python src/cnn_emotion.py
GPU auto-detected. Falls back to CPU if no GPU.
"""

import os, warnings, time
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
RAVDESS_PATH   = "data/Ravdess"
MODELS_DIR     = "models"
OUTPUTS_DIR    = "outputs"
SAMPLE_RATE    = 22050
N_MELS         = 128
TIME_FRAMES    = 128
BATCH_SIZE     = 32
MAX_EPOCHS     = 60
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 12      # early stopping
RANDOM_STATE   = 42
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15

RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm",    "03": "happy",    "04": "sad",
    "05": "angry",   "06": "fearful", "07": "disgust",  "08": "surprised"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────


def setup_dirs():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_mel(file_path: str) -> np.ndarray:
    """
    Returns fixed-size mel spectrogram (1, N_MELS, TIME_FRAMES).
    Consistent with what predict.py will use at inference.
    """
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Normalize amplitude
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # Trim leading/trailing silence
    signal, _ = librosa.effects.trim(signal, top_db=20)
    if len(signal) == 0:
        raise ValueError("Empty audio after trimming")

    # Mel spectrogram
    mel    = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=N_MELS,
        n_fft=2048, hop_length=512, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mn, mx = mel_db.min(), mel_db.max()
    mel_db = (mel_db - mn) / (mx - mn + 1e-8)

    # Pad or truncate to fixed time dimension
    t = mel_db.shape[1]
    if t < TIME_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, TIME_FRAMES - t)),
                        mode="constant", constant_values=0)
    else:
        mel_db = mel_db[:, :TIME_FRAMES]

    return mel_db[np.newaxis]    # (1, N_MELS, TIME_FRAMES)


# ─────────────────────────────────────────────────────────────
#  DATASET LOADING
# ─────────────────────────────────────────────────────────────
def load_ravdess() -> list:
    """
    Walks RAVDESS directory, extracts mel spectrograms.
    Returns list of (mel_array, emotion_label) tuples.
    Caches features to .npy for faster re-runs.
    """
    cache_x = os.path.join(MODELS_DIR, "ravdess_mel_x.npy")
    cache_y = os.path.join(MODELS_DIR, "ravdess_mel_y.npy")

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print("  Loading cached mel features...")
        X = np.load(cache_x, allow_pickle=True)
        Y = np.load(cache_y, allow_pickle=True)
        print(f"  Loaded {len(X)} cached samples.")
        return list(zip(X, Y))

    print(f"\n  Extracting mel spectrograms from RAVDESS...")
    print(f"  Path: {os.path.abspath(RAVDESS_PATH)}")

    samples, skipped = [], 0

    for root, _, files in os.walk(RAVDESS_PATH):
        for fname in sorted(files):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 7 or parts[0] != "03":   # speech only
                skipped += 1
                continue
            emotion = RAVDESS_EMOTION_MAP.get(parts[2])
            if not emotion:
                skipped += 1
                continue
            try:
                mel = extract_mel(os.path.join(root, fname))
                samples.append((mel, emotion))
                if len(samples) % 100 == 0:
                    print(f"    {len(samples)} files processed...")
            except Exception as e:
                skipped += 1

    print(f"  Done: {len(samples)} loaded, {skipped} skipped.")

    # Cache for next run
    np.save(cache_x, np.array([s[0] for s in samples], dtype=np.float32))
    np.save(cache_y, np.array([s[1] for s in samples]))
    print("  Features cached to models/ for faster re-runs.")

    return samples


# ─────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.LongTensor(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]

        # On-the-fly augmentation during training
        if self.augment:
            x = self._augment(x)

        return x, self.y[idx]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time/frequency masking (SpecAugment-lite).
        Improves generalization by randomly masking
        contiguous blocks of time or frequency.
        """
        x = x.clone()

        # Frequency masking — zero out random freq bands
        if torch.rand(1) > 0.5:
            f_start = torch.randint(0, N_MELS - 15, (1,)).item()
            f_end   = f_start + torch.randint(5, 20, (1,)).item()
            x[0, f_start:f_end, :] = 0

        # Time masking — zero out random time segments
        if torch.rand(1) > 0.5:
            t_start = torch.randint(0, TIME_FRAMES - 20, (1,)).item()
            t_end   = t_start + torch.randint(5, 25, (1,)).item()
            x[0, :, t_start:t_end] = 0

        return x


# ─────────────────────────────────────────────────────────────
#  CNN MODEL
# ─────────────────────────────────────────────────────────────
class EmotionCNN(nn.Module):
    """
    2D CNN on mel spectrograms.
    Input:  (batch, 1, 128, 128)
    Output: (batch, n_classes)

    Architecture uses residual-style skip connections in later
    blocks for better gradient flow with deep networks.
    """

    def __init__(self, n_classes: int = 8):
        super().__init__()

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # ── Block 1: 32→64, 128→64 ──
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # ── Block 2: 64→128, 64→32 ──
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # ── Block 3: 128→256, 32→16 ──
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        # ── Block 4: 256→512, 16→8 ──
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        # ── Global Average Pooling ──
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier ──
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────
#  CNN + LSTM HYBRID
# ─────────────────────────────────────────────────────────────
class EmotionCNNLSTM(nn.Module):
    """
    CNN extracts features per-frame, LSTM captures temporal
    dynamics. Bidirectional LSTM reads emotion progression
    both forward and backward for richer context.
    Expected: ~91% accuracy on RAVDESS.
    """

    def __init__(self, n_classes: int = 8,
                 lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()

        # CNN feature extractor (per time slice)
        self.cnn_feat = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 1)),          # pool freq, keep time
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, None)),  # keep time axis intact
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=256 * 8,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, freq, time)
        b, c, f, t = x.shape

        # CNN: process as (B, 1, freq, time) keeping time
        cnn_out = self.cnn_feat(x)       # (B, 256, 8, time)
        _, _, h, t2 = cnn_out.shape
        cnn_out = cnn_out.permute(0, 3, 1, 2)   # (B, time, 256, 8)
        cnn_out = cnn_out.contiguous().view(b, t2, -1)  # (B, time, 256*8)

        lstm_out, _ = self.lstm(cnn_out)
        # Use both first and last for global context
        out = torch.cat([lstm_out[:, 0, :], lstm_out[:, -1, :]], dim=1)

        # Handle bidirectional: lstm_out has 2*hidden dim already
        out = lstm_out[:, -1, :]    # last time step (B, 2*hidden)
        return self.head(out)


# ─────────────────────────────────────────────────────────────
#  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────
def compute_class_weights(y_train: np.ndarray,
                          n_classes: int) -> torch.Tensor:
    counts  = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights).to(DEVICE)


def train_one_epoch(model, loader, optimizer,
                    criterion, scaler_amp) -> tuple:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision (faster on GPU)
        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                out  = model(X)
                loss = criterion(out, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        correct    += out.argmax(1).eq(y).sum().item()
        total      += y.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion) -> tuple:
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    for X, y in loader:
        X, y  = X.to(DEVICE), y.to(DEVICE)
        out   = model(X)
        loss  = criterion(out, y)
        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="weighted", zero_division=0)
    return total_loss / len(loader), acc, f1, preds, trues


# ─────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────
def save_training_plots(train_acc, val_acc, train_loss,
                        val_loss, tag: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(train_acc,  color="#1D9E75", lw=2, label="Train")
    ax1.plot(val_acc,    color="#E24B4A", lw=2, label="Val",
             linestyle="--")
    ax1.set(title=f"{tag} — Accuracy", xlabel="Epoch",
            ylabel="Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(train_loss, color="#1D9E75", lw=2, label="Train")
    ax2.plot(val_loss,   color="#E24B4A", lw=2, label="Val",
             linestyle="--")
    ax2.set(title=f"{tag} — Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR,
                        f"training_{tag.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


def save_confusion_matrix(y_true, y_pred, le, tag: str):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.4, linecolor="white", ax=ax)
    ax.set_title(f"{tag} Confusion Matrix (%)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR,
                        f"cm_{tag.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────
#  MAIN TRAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def train(model_type: str = "cnn",
          extra_samples: list = None) -> dict:
    """
    Full training pipeline.
    model_type: "cnn" or "cnn_lstm"
    extra_samples: list of (mel, label) from IEMOCAP or other datasets
    Returns dict with accuracy, f1, model, le.
    """
    setup_dirs()
    tag = "CNN" if model_type == "cnn" else "CNN+LSTM"

    print("\n" + "=" * 58)
    print(f"  TRAINING {tag} EMOTION MODEL")
    print(f"  Device : {DEVICE}")
    print(f"  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}")
    print("=" * 58)

    # ── Load data ─────────────────────────────────────────────
    samples = load_ravdess()

    if extra_samples:
        samples.extend(extra_samples)
        print(f"\n  Combined: {len(samples)} total samples")

    if not samples:
        print("  ERROR: No samples found. Check RAVDESS path.")
        return {}

    X_all = np.array([s[0] for s in samples], dtype=np.float32)
    y_raw = [s[1] for s in samples]

    # ── Encode labels ──────────────────────────────────────────
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    print(f"\n  Classes ({n_classes}): {list(le.classes_)}")
    print("  Distribution:")
    for i, cls in enumerate(le.classes_):
        cnt = (y_all == i).sum()
        bar = "█" * (cnt // 20)
        print(f"    {cls:<12} {bar} ({cnt})")

    # ── Split: 70% train / 15% val / 15% test ─────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_all)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_tmp)

    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Weighted sampler to handle class imbalance ─────────────
    class_counts = np.bincount(y_train, minlength=n_classes)
    sample_weights = 1.0 / (class_counts[y_train] + 1e-8)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(y_train), replacement=True
    )

    # ── DataLoaders ────────────────────────────────────────────
    train_ds = MelDataset(X_train, y_train, augment=True)
    val_ds   = MelDataset(X_val,   y_val,   augment=False)
    test_ds  = MelDataset(X_test,  y_test,  augment=False)

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=nw,
                              pin_memory=DEVICE.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=nw)

    # ── Model, loss, optimizer ─────────────────────────────────
    model = (EmotionCNN(n_classes) if model_type == "cnn"
             else EmotionCNNLSTM(n_classes)).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_p:,}")

    cw        = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                             lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    # Mixed precision scaler (GPU only)
    amp_scaler = (torch.cuda.amp.GradScaler()
                  if DEVICE.type == "cuda" else None)

    # ── Training loop ──────────────────────────────────────────
    best_val_acc = 0.0
    pat_cnt      = 0
    best_path    = os.path.join(MODELS_DIR, f"cnn_emotion_best.pt")

    ta_hist, va_hist, tl_hist, vl_hist = [], [], [], []

    print(f"\n  {'Ep':>3} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VaLoss':>8} {'VaAcc':>7} {'LR':>9} {'Best':>6}")
    print("  " + "─" * 56)

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, amp_scaler)
        va_loss, va_acc, va_f1, _, _ = evaluate_model(
            model, val_loader, criterion)
        scheduler.step()

        ta_hist.append(tr_acc); tl_hist.append(tr_loss)
        va_hist.append(va_acc); vl_hist.append(va_loss)

        lr  = optimizer.param_groups[0]["lr"]
        new = "★" if va_acc > best_val_acc else " "

        print(f"  {epoch:>3} {tr_loss:>8.4f} {tr_acc*100:>6.2f}% "
              f"{va_loss:>8.4f} {va_acc*100:>6.2f}% "
              f"{lr:>9.6f} {new}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            pat_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     va_acc,
                "n_classes":   n_classes,
                "model_type":  model_type,
                "le_classes":  list(le.classes_),
            }, best_path)
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(patience={PATIENCE})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/60:.1f} minutes")

    # ── Test evaluation ────────────────────────────────────────
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc, test_f1, y_pred, y_true = evaluate_model(
        model, test_loader, criterion)

    print(f"\n  {'='*55}")
    print(f"  FINAL TEST RESULTS — {tag}")
    print(f"  {'='*55}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test F1 Score : {test_f1*100:.2f}%")
    print(f"  Best Val Acc  : {best_val_acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=le.classes_,
                                 digits=3))

    # ── Save plots and label encoder ──────────────────────────
    save_training_plots(ta_hist, va_hist, tl_hist, vl_hist, tag)
    save_confusion_matrix(y_true, y_pred, le, tag)
    joblib.dump(le, os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))

    print(f"\n  Saved: models/cnn_emotion_best.pt")
    print(f"  Saved: models/cnn_label_encoder.pkl")

    return {
        "model":     model,
        "le":        le,
        "accuracy":  test_acc,
        "f1":        test_f1,
        "tag":       tag,
        "best_path": best_path,
    }


# ─────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────
def load_cnn_for_inference():
    """Load trained CNN model for use in predict.py."""
    path = os.path.join(MODELS_DIR, "cnn_emotion_best.pt")
    le_p = os.path.join(MODELS_DIR, "cnn_label_encoder.pkl")

    if not os.path.exists(path):
        return None, None

    ckpt  = torch.load(path, map_location=DEVICE)
    mtype = ckpt.get("model_type", "cnn")
    nc    = ckpt["n_classes"]
    model = EmotionCNN(nc) if mtype == "cnn" else EmotionCNNLSTM(nc)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    le = joblib.load(le_p) if os.path.exists(le_p) else None
    return model, le


@torch.no_grad()
def predict_single(file_path: str,
                   model=None, le=None) -> tuple:
    """
    Predict emotion from one audio file using CNN.
    Returns (emotion_str, confidence_float, all_probs_dict).
    """
    if model is None:
        model, le = load_cnn_for_inference()
        if model is None:
            return "neutral", 0.0, {}

    try:
        mel    = extract_mel(file_path)
        x      = torch.FloatTensor(mel).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
        conf   = float(probs[idx]) * 100
        emo    = le.inverse_transform([idx])[0]
        all_p  = {le.classes_[i]: round(float(probs[i])*100, 2)
                  for i in range(len(le.classes_))}
        return emo, round(conf, 1), all_p
    except Exception as e:
        print(f"  CNN predict error: {e}")
        return "neutral", 0.0, {}


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 58)
    print("  CNN EMOTION CLASSIFIER")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM   : {mem / 1e9:.1f} GB")
    print("=" * 58)

    print("\n  1. Train CNN on RAVDESS (Day 1 — target 85%+)")
    print("  2. Train CNN+LSTM on RAVDESS (Day 1 — target 91%+)")
    print("  3. Predict from audio file")
    print("  4. Exit")

    choice = input("\n  Choice: ").strip()

    if choice == "1":
        result = train(model_type="cnn")
        if result:
            print(f"\n  Done. CNN accuracy: {result['accuracy']*100:.2f}%")

    elif choice == "2":
        result = train(model_type="cnn_lstm")
        if result:
            print(f"\n  Done. CNN+LSTM accuracy: {result['accuracy']*100:.2f}%")

    elif choice == "3":
        path = input("  Audio file: ").strip().strip('"')
        if os.path.exists(path):
            emo, conf, probs = predict_single(path)
            print(f"\n  Emotion    : {emo}")
            print(f"  Confidence : {conf:.1f}%")
            print("\n  All probabilities:")
            for e, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(p / 5)
                print(f"    {e:<12} {bar} ({p:.1f}%)"
                      + (" <--" if e == emo else ""))
        else:
            print("  File not found.")


if __name__ == "__main__":
    main()