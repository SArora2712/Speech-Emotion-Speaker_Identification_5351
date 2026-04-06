import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

#  CONFIGURA
dataset_path = "data\Custom_Dataset"
csv_output="data\custom_features.csv"
N_mfcc=40
sample_rate=22050

valid_emotions=[
    
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

def get_emotion_from_path(root):
    parts=root.replace("\\","/").split("/")
    return parts[-1].lower()

def get_speaker_from_path(root):
    parts=root.replace("\\","/").split("/")
    return parts[-2]

def preprocess_audio(file_path):
    signal,sr=librosa.load(file_path,sr=sample_rate)

    if np.max(np.abs(signal))>0:
        signal=signal/np.max(np.abs(signal))
    
    
    signal,_=librosa.effects.trim(signal,top_db=20)

    return signal,sr

def extract_mfcc_features(signal,sr):
    mfcc=librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=N_mfcc)
    return np.mean(mfcc,axis=1 )

def build_custom_dataset():
    rows=[]
    total=0
    skipped=0

    print("\n"+"-"*50)
    print("Extracting features from audio files...")
    print(f"Looking in : {os.path.abspath(dataset_path)}")
    print("-"*50)

    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".wav"):
                continue

            emotion=get_emotion_from_path(root)
            speaker=get_speaker_from_path(root)

            if emotion not in valid_emotions:
                print(f'Skipping{file} -\'{emotion}\' not in Ravdesss emotions')
                skipped+=1
                continue

            file_path=os.path.join(root,file)

            try:
                signal,sr=preprocess_audio(file_path)
                mfcc=extract_mfcc_features(signal,sr)

                row={f"mfcc_{i+1}":mfcc[i] for i in range(N_mfcc)}
                row["emotion"]=emotion
                row["speaker"]=speaker
                row["file"]=file
                rows.append(row)
                total+=1

                if total%50==0:
                    print(f"Processed{total} files.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                skipped+=1  

    if total==0:
        print("Error:No files processed.\nPlease check the dataset path and structure.")
        return None
            
    print(f"\n Done Processing. \n \tTotal: {total}\n \t Skipped: {skipped}")
    return pd.DataFrame(rows)

def show_summary(df):
    print("\n"+"-"*50)
    print("\nDataset Summary:")
    print("-"*50)
    print(f"Total samples: {len(df)}")
    print(f"Featurs or Samples:{N_mfcc} MFCC cefficients")
    print(f"Unique emotions: {df['emotion'].nunique()}")
    print(f"Unique speakers: {df['speaker'].nunique()}")

    print("\n Samples per speaker:")
    for speaker , count in df['speaker'].value_counts().items():
        bar="█" *(count//10)
        print(f"\t{speaker}: {bar} {count} samples")

    print("\n Samples per emotion:")
    for emotion,count in df["emotion"].value_counts().items():
        bar="█"*(count//10)
        print(f"\t{emotion:<12}: {bar} {count} samples")

def visualize_speaker_distribution(df):
    
    speaker_counts=df["speaker"].value_counts()

    plt.figure(figsize=(10,6))
    bars=plt.bar(
        speaker_counts.index,
        speaker_counts.values,
        color="#1D9E75",
        edgecolor="#0F6E56",
        linewidth=0.5
    )

    for bar, val in zip(bars,speaker_counts.values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()+0.3,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10
        )
    plt.title("Custom Dataset-Recordings per Speaker",fontsize=14,fontweight="bold")
    plt.xlabel("Speaker ID",fontsize=12)
    plt.ylabel("Number of Recordings",fontsize=12)
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.savefig("speaker_distribution.png",dpi=300)
    plt.show()
    print("Saved -> Custom_speaker_distribution.png")

def visualize_emoton_dist(df):
    emotion_counts=df["emotion"].value_counts()

    colors={
        "neutral":"#888780",
        "happy":"#1D9E75",
        "sad":"#EF9F27",
        "angry":"#378ADD",
        "fearful":"#E24B4A",
        "disgust":"#D85A30",
        "surprised":"#D4537E"
    }
    bar_colors=[colors.get(e,"#888780") for e in emotion_counts.index]

    plt.figure(figsize=(10,5))
    bars=plt.bar(
        emotion_counts.index,
        emotion_counts.values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5
    )

    for bar,val in zip(bars,emotion_counts.values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()+0.3,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.title("Custom Dataset-Recordings per Emotion",fontsize=14,fontweight="bold")
    plt.xlabel("Emotion",fontsize=12)
    plt.ylabel("Number of Recordings",fontsize=12)
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.savefig("custom_emotion_distribution.png",dpi=300)
    plt.show()  
    print("Saved -> Custom_emotion_distribution.png")

def visualize_heatmap(df):
    emotions=sorted(df["emotion"].unique())
    cols=4
    rows_n=(len(emotions)+cols-1)//cols

    fig,axes=plt.subplots(rows_n,cols,figsize=(16,4*rows_n))
    axes=axes.flatten()

    mfcc_cols=[f"mfcc_{i+1}" for i in range(N_mfcc)]

    for i,emotion in enumerate(emotions):
        subset=df[df["emotion"]==emotion][mfcc_cols].values
        mean_mfcc=np.mean(subset,axis=0).reshape(1,-1)

        axes[i].imshow(mean_mfcc,cmap="viridis",aspect="auto")
        axes[i].set_title(f"Mean MFCC - {emotion}",fontsize=12,fontweight="bold")   
        axes[i].set_xlabel("MFCC Coefficients",fontsize=10)
        axes[i].set_yticks([])
    for j in range(i+1,len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Custom Dataset - Mean MFCC Heatmaps by Emotion",fontsize=16,fontweight="bold")
    plt.tight_layout()
    plt.savefig("custom_mfcc_heatmaps.png",dpi=300)
    plt.show()
    print("Saved -> Custom_mfcc_heatmaps.png")

def main():
    # 1-> Extract features and build dataset
    df=build_custom_dataset()
    if df is None:
        return
    
    # 2-> Show summary
    show_summary(df)    

    #3-> Save to CSV
    df.to_csv(csv_output,index=False)
    print(f"\nDataset saved to {csv_output}")
    print(f"CSV file contains {len(df)} samples with {N_mfcc} MFCC features each.")
    print(f"Shape of dataset: {df.shape}")

    #4-> Visualize distributions
    print("\nVisualizing speaker distribution...")
    visualize_speaker_distribution(df)
    visualize_emoton_dist(df)
    visualize_heatmap(df)

    print("\n"+"-"*50)
    print("Complete! Custom dataset features extracted, saved, and visualized.")
    print("Files generated:\n\t- custom_features.csv\n\t- custom_speaker_distribution.png\n\t- custom_emotion_distribution.png\n\t- custom_mfcc_heatmaps.png")
    print("-"*50)
    
if __name__=="__main__":

    main()

