import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,classification_report,
    confusion_matrix,f1_score
    )

csv_path="mfcc_features.csv"
models_folder="saved_models"
Random_state=42
Test_size=0.2
N_mfcc=40

def create_folder(path):
    os.makedirs(path,exist_ok=True)

def load_data():
    print("\n"+"-"*50)
    print("Loading ravdess Feature dataset")
    print("-"*50)

    if not os.path.exists(csv_path):
        print(f"Error:{csv_path} not found!")
        print("Run feature_extraction.py first")
        return None,None,None,None
    
    df=pd.read_csv(csv_path)
    print(f"Loaded : {df.shape[0]} samples x {df.shape[1]} columns")
    print(f"Emotions: {sorted(df['emotion'].unique())}")
    print(f"Samples Per Emotion:")
    for emo,count in df["emotion"].value_counts().items():
        bar="█" * (count//20)
        print(f"{emo:<12} {bar} ({count})")

    mfcc_cols = [col for col in df.columns if "mfcc" in col.lower()]
    x=df[mfcc_cols].values
    y=df["emotion"].values

    return x,y,df,mfcc_cols

def preprocess(x,y):
    print("\n Prepocessing...")

    le=LabelEncoder()
    y_encoded=le.fit_transform(y)
    print("Label mapping:")
    for i, label in enumerate(le.classes_):
        print(f"{label} → {i}")

    #Train Test Split
    x_train,x_test,y_train,y_test=train_test_split(
        x,y_encoded,
        test_size=Test_size,
        random_state=Random_state,
        stratify=y_encoded
    )

    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    print(f"Training Samples: {x_train.shape[0]}")
    print(f"Testing smaples : {x_test.shape[0]}")

    return x_train,x_test,y_train,y_test,le,scaler

def train_svm(x_train,y_train):
    print("\n Training SVM")
    model=SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=Random_state
    )
    model.fit(x_train,y_train)
    print("SVM training complete")
    return model

def train_random_forest(x_train,y_train):
    print("\n Traning Random Forest")
    model=RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=Random_state,
        n_jobs=-1
    )
    model.fit(x_train,y_train)
    print("Random Forest traiing complete")
    return model


def train_logisticRegresssion(x_train,y_train):
    print("\nTraining Logistic Regresssion")
    model=LogisticRegression(
        max_iter=1000,
        random_state=Random_state,
        
    )
    model.fit(x_train,y_train)
    print("Logistic Regression training complete")
    return model

def train_xgboost(x_train,y_train):
    print("\n Training XGBoost")
    model=XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=Random_state,
        eval_metric="mlogloss"
    )
    model.fit(x_train,y_train)
    print("XGBoost training complete")
    return model

def train_mlpclassifier(x_train,y_train):
    print("\n Training MLP Classifier")
    model=MLPClassifier(
        hidden_layer_sizes=(256,128,64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=Random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(x_train,y_train)
    print("MLP Classifier training complete")
    return model

def evaluate_model(model,x_test,y_test,le,model_name):
    print(f"\n {'-'*50}")
    print(f"Resuts-{model_name}")
    
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred,average='weighted')

    print(f'Accuracy : {accuracy*100:.2f}%')
    print(f'F1 Score : {f1*100:.2f}%')
    print(f"Classificatio Report: ")
    print(classification_report(
        y_test,y_pred,
        target_names=le.classes_,
        digits=3
    ))

    return accuracy,f1,y_pred

def confusion_matrix_vis(y_test,y_pred,le,model_name):
    cm=confusion_matrix(y_test,y_pred)
    cm_pct=cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]*100
    labels=le.classes_
    fname=f"confusion_matrix_{model_name.lower().replace(' ','_')}.png"

    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label":"Percentage(%)"}
        
    )
    plt.title(f"Confusion Matrix-{model_name}\n(values in % )",fontsize=14,fontweight="bold")
    plt.ylabel("Actual Emotion",fontsize=12)
    plt.xlabel("Predicted Emotion",fontsize=12)
    plt.xticks(rotation=45,ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname,dpi=150)
    plt.show()
    print(f"Saved -> {fname}")
    return fname

def plot_accuracy_comp(results):
    models=list(results.keys())
    accuracies=[results[m]['Accuracy']*100 for m in models]
    f1_scores=[results[m]['F1 Score']*100 for m in models] 

    x=np.arange(len(models))
    width=0.35

    fig,ax=plt.subplots(figsize=(9,5))
    bars1=ax.bar(x-width/2,accuracies,width,label="Accuracy",
                 color='#1D9E75',edgecolor="white")
    bars2=ax.bar(x+width/2,f1_scores,width,label="F1 Score",
                 color="#378ADD",edgecolor="white")
    
    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                f"{bar.get_height():.1f}%",ha="center",va="bottom",fontsize=10,color='#1D9E75')
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                f"{bar.get_height():.1f}%",ha="center",va="bottom",fontsize=10,color="#378ADD")
        
    ax.set_xlabel("Model")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison - Accuracy & F1 Score ",fontsize=14,fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0,110)
    ax.legend()
    ax.grid(axis="y",alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
    print("Saved \"model_comparison.png\"")

def save_models(best_model,le,scaler,best_model_name):
    create_folder(models_folder)

    model_path=os.path.join(models_folder,"emotion_model.pkl")
    le_path=os.path.join(models_folder,"label_encoder_emotion.pkl")
    scaler_path=os.path.join(models_folder,"scaler_emotion.pkl")
    name_path=os.path.join(models_folder,"best_model_name.txt")

    joblib.dump(best_model,model_path)
    joblib.dump(le,le_path)
    joblib.dump(scaler,scaler_path)

    with open(name_path,"w") as f:
        f.write(best_model_name)

    print(f"Models saved to \'{models_folder}\'")
    print(f"\temotion_model.pkl")
    print(f'\tlabel_encoder_emotion.pkl')
    print(f"\tscaler-emotion.pkl")

def main():
    print("\n"+"-"*50)
    print( "EMOTION MODEL TRAINING")
    print("Datasets: ravdess| Features:40 MFCC")
    print("-"*55)

    #1-> Load data
    x, y, df, mfcc_cols = load_data()
    if x is None :
        return 
    
    # 2-> Preprocess
    x_train,x_test,y_train,y_test,le,scaler=preprocess(x,y)

    # 3-> Train Models
    models={
        "SVM":train_svm(x_train,y_train),
        "Random Forest":train_random_forest(x_train,y_train),
        "Logistic Regression": train_logisticRegresssion(x_train,y_train),
        "XGBoost": train_xgboost(x_train,y_train),
        "MLP Classifier": train_mlpclassifier(x_train,y_train)

    }

    # 4-> Evaluate Models
    print('\n'+"-"*55)
    print("Evaluation Ressults")

    results={}
    all_y_pred={}

    for model_name,model in models.items():
        accuracy,f1,y_pred=evaluate_model(
            model,x_test,y_test,le,model_name
        )
        results[model_name]={"Accuracy":accuracy,'F1 Score':f1,"Model":model}
        all_y_pred[model_name]=y_pred

    # 5-> Find Best model
    best_model_name=max(results,key=lambda m: results[m]["Accuracy"])
    best_model=results[best_model_name]['Model']
    best_accuracy=results[best_model_name]['Accuracy']
    best_f1=results[best_model_name]['F1 Score']

    print("\n"+"-"*50)
    print("Best Model")
    print("-"*50)
    print(f"Model: {best_model_name}")
    print(f" Accuracy: {best_accuracy*100:.2f}%")
    print((f' F1 Score : {best_f1*100:.2f}%'))

    # 6-> Plot confusion matrix
    print("\n Confusion matrix")
    confusion_matrix_vis(
        y_test,
        all_y_pred[best_model_name],
        le,
        best_model_name
    )

    # 7-> Plot mOdel Comparison
    print("\n Model Comparison Chart ")
    plot_accuracy_comp(results)

    # 8-> Save best model
    save_models(best_model,le,scaler,best_model_name)

    # 9-> Final Summary
    print("\n"+"-"*50)
    print("Training Complete")
    print("-"*50)
    print("  TRAINING COMPLETE — SUMMARY")
    print("-" * 50)
    print(f"  Total samples trained on : {x_train.shape[0]}")
    print(f"  Total samples tested on  : {x_test.shape[0]}")
    print(f"  Best model               : {best_model_name}")
    print(f"  Best accuracy            : {best_accuracy*100:.2f}%")
    print(f"  Best F1 score            : {best_f1*100:.2f}%")
    print(f"\n  Files saved:")
    print(f"    saved_models/emotion_model.pkl")
    print(f"    saved_models/label_encoder_emotion.pkl")
    print(f"    saved_models/scaler_emotion.pkl")
    print(f"    confusion_matrix_{best_model_name.lower().replace(' ', '_')}.png")
    print(f"    model_comparison.png")
    print("-" * 55)


if __name__=="__main__":
    main()









