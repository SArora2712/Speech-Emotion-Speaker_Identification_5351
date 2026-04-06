import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    print("XGBoost is not installed. Skipping XGBClassifier.")
    xgboost_available = False

csv_path = 'data/custom_features.csv'
models_folder = 'models'
Random_state = 42
test_size = 0.2
N_mfcc = 40

def create_models_folder(path):
    os.makedirs(path, exist_ok=True)

def load_data():
    print("\n"+"-"*50)
    print(f"Custom Dataset: Loading data from {os.path.abspath(csv_path)}")
    print("-"*50)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run Custom_data_features.py to generate the dataset.")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully with {len(df)} samples.")
    print(f"Loaded: {df.shape[0]} samples x {df.shape[1]} features. ")    
    print(f"Speakers: {df['speaker'].nunique()} speakers")
    print(f"\n Samples per Speaker:")
    for speaker, count in df['speaker'].value_counts().items():
        bar = '█' * (count // 2)  # Scale the bar length for better visualization
        print(f" {speaker:<14} {bar} ({count})")

    print("\n Samples per emotion:")
    for emotion, count in df['emotion'].value_counts().items():
        bar = '█' * (count // 2)  # Scale the bar length for better visualization
        print(f" {emotion:<12} {bar} ({count})")

    mfcc_cols = [f"mfcc_{i}" for i in range(1,N_mfcc+1)]
    x = df[mfcc_cols].values
    y = df['speaker'].values

    return x, y, df

def preprocess_data(x, y):
    print("\n Preprocessing data...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Speker Encoding:")
    for speaker, code in zip(le.classes_, le.transform(le.classes_)):
        print(f" {speaker} → {code}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, 
        test_size=test_size, 
        random_state=Random_state,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(f"Data split: {len(x_train)} training samples, {len(x_test)} testing samples.")
    return x_train, x_test, y_train, y_test, le, scaler

def train_svm(x_train, y_train):
    print("\n Training SVM...")

    
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=Random_state
    )
    model.fit(x_train, y_train)
    print(f"SVM trained with kernel={model.kernel}, C={model.C}, gamma={model.gamma}.")
    return model

def train_random_forest(x_train, y_train):
    print("\n Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=Random_state,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    print(f"Random Forest trained with n_estimators={model.n_estimators}, max_depth={model.max_depth}.")
    return model

def train_logistic_regression(x_train, y_train):
    print("\n Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=Random_state,
        solver="lbfgs"
    )
    model.fit(x_train, y_train)
    print(f"Logistic Regression trained with C={model.C}, max_iter={model.max_iter}.")
    return model

def train_xgboost(x_train, y_train):
    
    print("\n Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        sub_sample=0.8,
        colsample_bytree=0.8,
        random_state=Random_state,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit( x_train, y_train)
    print(f"XGBoost trained with n_estimators={model.n_estimators}, max_depth={model.max_depth}.")
    return model

def train_mlp(x_train, y_train):
    print("\n Training MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(256,128,64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=Random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True
    )
    model.fit(x_train, y_train)
    print(f"MLP trained with hidden_layer_sizes={model.hidden_layer_sizes}, activation={model.activation}.")
    return model

def evaluate_model(model, x_test, y_test, le, model_name):
    print(f"\n"+"-"*50)
    print(f"Evaluating {model_name}...")
    print("-"*50)

    y_pred=model.predict(x_test)
    acuracy=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred,average="weighted")

    print(f"Accuracy: {acuracy*100:.2f}%")
    print(f"F1-score: {f1*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test,y_pred,
        target_names=le.classes_,
        digits=3))
    
    return acuracy, f1,y_pred

def plot_confusion_matrix(y_test, y_pred, le, model_name):
    print(f"\n Plotting confusion matrix for {model_name}...")
    cm=confusion_matrix(y_test,y_pred)
    cm_pct=cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]*100
    labels=le.classes_
    fname=f"outputs/confusion_matrix_{model_name.lower().replace(' ','_')}.png"
    
    fig_size=max(12,len(labels))
    plt.figure(figsize=(fig_size,fig_size-1))
    sns.heatmap(
            cm_pct, 
            annot=cm, 
            fmt=".1f", 
            cmap="Purples",
            xticklabels=labels, 
            yticklabels=labels,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Percentage (%)"}
        )
    plt.title(
    f"Speaker Identification — Confusion Matrix\n"
    f"{model_name} (values in %)",
    fontsize=14, fontweight="bold"
    )
    plt.ylabel("Actual Speaker", fontsize=12)
    plt.xlabel("Predicted Speaker", fontsize=12)
    plt.xticks(rotation=45, ha="right")        
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved -> {fname}")

def plot_model_comparison(results):
    models     = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    f1_scores  = [results[m]["f1"] * 100 for m in models]
 
    x     = np.arange(len(models))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accuracies, width,
                   label="Accuracy", color="#7F77DD", edgecolor="white")
    bars2 = ax.bar(x + width/2, f1_scores, width,
                   label="F1 Score", color="#D4537E", edgecolor="white")
 
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=9
        )
 
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(
        "Speaker Identification — All Models Comparison",
        fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs\speaker_model_comparison.png", dpi=150)
    plt.show()
    print("  Saved -> speaker_model_comparison.png")

def plot_per_speaker_accuracy(y_test, y_pred, le, model_name):
    cm          = confusion_matrix(y_test, y_pred)
    per_speaker = cm.diagonal() / cm.sum(axis=1) * 100
    labels      = le.classes_
 
    colors = [
        "#1D9E75" if acc >= 80 else
        "#EF9F27" if acc >= 60 else
        "#E24B4A"
        for acc in per_speaker
    ]
 
    plt.figure(figsize=(14, 5))
    bars = plt.bar(labels, per_speaker, color=colors, edgecolor="white")
 
    for bar, val in zip(bars, per_speaker):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )
    plt.axhline(y=80, color="green", linestyle="--",
                linewidth=1, alpha=0.6, label="80% threshold")
    plt.axhline(y=60, color="orange", linestyle="--",
                linewidth=1, alpha=0.6, label="60% threshold")
 
    plt.title(
        f"Speaker Identification Accuracy per Speaker — {model_name}",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Speaker", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 115)
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs\per_speaker_accuracy.png", dpi=150)
    plt.show()
    print("  Saved -> per_speaker_accuracy.png")

def plot_mlp_loss(mlp_model):
    if not hasattr(mlp_model, "loss_curve_"):
        return
 
    plt.figure(figsize=(9, 4))
    plt.plot(mlp_model.loss_curve_, color="#7F77DD",
             linewidth=2, label="Training loss")
 
    if (hasattr(mlp_model, "validation_scores_") and
            mlp_model.validation_scores_):
        val_loss = [1 - s for s in mlp_model.validation_scores_]
        plt.plot(val_loss, color="#D4537E", linewidth=2,
                 linestyle="--", label="Validation loss")
 
    plt.title("MLP Neural Network — Training Loss Curve",
              fontsize=14, fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs\mlp_loss_curve_speaker.png", dpi=150)
    plt.show()
    print("  Saved -> mlp_loss_curve_speaker.png")
 
 
def save_models(best_model, le, scaler, best_model_name):
    create_models_folder(models_folder)
 
    joblib.dump(best_model,
                os.path.join(models_folder, "speaker_model.pkl"))
    joblib.dump(le,
                os.path.join(models_folder, "label_encoder_speaker.pkl"))
    joblib.dump(scaler,
                os.path.join(models_folder  , "scaler_speaker.pkl"))
 
    with open(os.path.join(models_folder,
              "best_speaker_model_name.txt"), "w") as f:
        f.write(best_model_name)
 

    print(f"\n  Saved to '{models_folder }/':")
    print(f"    speaker_model.pkl          <- {best_model_name}")
    print(f"    label_encoder_speaker.pkl")
    print(f"    scaler_speaker.pkl")
 
def main():
    print("\n" + "=" * 55)
    print("  SPEAKER IDENTIFICATION MODEL TRAINING")
    print("  Dataset  : Custom Dataset")
    print("  Models   : SVM, Random Forest, Logistic")
    print("             Regression, XGBoost, MLP")
    print("  Features : 40 MFCC coefficients")
    print("=" * 55)
 
    # ── Step 1: Load ──────────────────────────────────────────
    x, y, df = load_data()
    if x is None:
        return
 
    # ── Step 2: Preprocess ────────────────────────────────────
    x_train, x_test, y_train, y_test, le, scaler = preprocess_data(x, y)
 
    # ── Step 3: Train all models ──────────────────────────────
    print("\n" + "=" * 55)
    print("  TRAINING ALL MODELS")
    print("=" * 55)
    
    models = {}
    models["SVM"]                 = train_svm(x_train, y_train)
    models["Random Forest"]       = train_random_forest(x_train, y_train)
    models["Logistic Regression"] = train_logistic_regression(x_train, y_train)
 
    if xgboost_available:
        models["XGBoost"] = train_xgboost(x_train, y_train)
    else:
        print("\n  [4/5] XGBoost skipped — run: pip install xgboost")
 
    models["MLP Neural Network"] = train_mlp(x_train, y_train)
 
    # ── Step 4: Evaluate ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
 
    results    = {}
    all_y_pred = {}
 
    for model_name, model in models.items():
        accuracy, f1, y_pred = evaluate_model(
            model, x_test, y_test, le, model_name
        )
        results[model_name]    = {"accuracy": accuracy, "f1": f1,
                                   "model": model}
        all_y_pred[model_name] = y_pred
 
    # ── Step 5: Best model ────────────────────────────────────
    best_model_name = max(results, key=lambda m: results[m]["accuracy"])
    best_model      = results[best_model_name]["model"]
    best_accuracy   = results[best_model_name]["accuracy"]
    best_f1         = results[best_model_name]["f1"]
    best_y_pred     = all_y_pred[best_model_name]
 
    print("\n" + "=" * 55)
    print("  BEST MODEL")
    print("=" * 55)
    print(f"  Model    : {best_model_name}")
    print(f"  Accuracy : {best_accuracy*100:.2f}%")
    print(f"  F1 Score : {best_f1*100:.2f}%")
 
    # ── Step 6: Visualizations ───────────────────────────────
    print("\n  Generating confusion matrix...")
    plot_confusion_matrix(y_test, best_y_pred, le, best_model_name)
 
    print("\n  Generating model comparison chart...")
    plot_model_comparison(results)
 
    print("\n  Generating per-speaker accuracy chart...")
    plot_per_speaker_accuracy(y_test, best_y_pred, le, best_model_name)
 
    print("\n  Generating MLP loss curve...")
    plot_mlp_loss(models["MLP Neural Network"])
 
    # ── Step 7: Save ─────────────────────────────────────────
    save_models(best_model, le, scaler, best_model_name)
 
    # ── Step 8: Summary ──────────────────────────────────────
    print("\n" + "-" * 55)
    print("  TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 55)
    print(f"  Training samples : {x_train.shape[0]}")
    print(f"  Testing samples  : {x_test.shape[0]}")
    print(f"  Total speakers   : {len(le.classes_)}")
 
    print(f"\n  All model results (ranked):")
    for name, res in sorted(
        results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    ):
        marker = "  <-- BEST" if name == best_model_name else ""
        print(f"    {name:<22} {res['accuracy']*100:.2f}%"
              f"  F1: {res['f1']*100:.2f}%{marker}")
 
    print(f"\n  Best model  : {best_model_name}")
    print(f"  Accuracy    : {best_accuracy*100:.2f}%")
    print(f"  F1 Score    : {best_f1*100:.2f}%")
    print(f"\n  Files saved:")
    print(f"    saved_models/speaker_model.pkl")
    print(f"    saved_models/label_encoder_speaker.pkl")
    print(f"    saved_models/scaler_speaker.pkl")
    print(f"    confusion_matrix_speaker_*.png")
    print(f"    speaker_model_comparison.png")
    print(f"    per_speaker_accuracy.png")
    print(f"    mlp_loss_curve_speaker.png")
    print("-" * 55)
 
 
if __name__ == "__main__":
    main()
  
