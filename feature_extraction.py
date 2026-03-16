import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd

###Coonfiguration

dataset_path="Ravdess"
csv_output="mfcc_features.csv"
N_MFCC=40 # No of MFCC coefficients to extract
sample_rate=22050 # Hertz

# Ravdess labels for emotions
emotion_map={
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

def parse_filename(file_name):
    """
    Ravdess file-> [modality]-[channel]-[emotion]-[intensity]-[statement]-[repetition]-[actor]
    """
    parts=file_name.replace(".wav","").split("-")
    modality_code=parts[0]  # speech=01  and song=02
    emotion_code=parts[2]
    actor_id=parts[6]

    modality="speech" if modality_code=="01" else "song"
    emotion=emotion_map.get(emotion_code,"unknown")
    return modality,emotion,actor_id

def extract_mfcc(file_path,n_mfcc=40) :
    signal,sr=librosa.load(file_path,sr=sample_rate)
    
    #Auto trimming or stripping silence
    signal,_=librosa.effects.trim(signal,top_db=20) 

    # MFCC Extarction
    mfccs=librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=n_mfcc)

    #MEan across time axis -> shape (n_mfcc,)
    mfcc_mean=np.mean(mfccs,axis=1)

    return mfcc_mean
    

def build_dataset():
    rows=[]
    total=0
    skipped=0

    print("\n"+"="*55)
    print(" Extarcting MFCC features from Ravdess dataset ... ")
    print("="*55)

    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            if not file.endswith("wav"):
                continue

            modality,emotion,actor_id=parse_filename(file)

            parts = file.replace(".wav", "").split("-")
            if parts[0] != "03":               # ← USE THIS INSTEAD
                skipped += 1
                continue
            file_path=os.path.join(root,file)

            try:
                mfccs=extract_mfcc(file_path,n_mfcc=N_MFCC)
                
                row={f"mfcc-_{i+1}": mfccs[i] for i in range(N_MFCC)}
                row["emotion"]=emotion
                row["actor_id"]=actor_id
                row["file"]=file

                rows.append(row)
                total+=1

                if total % 50 ==0:
                    print(f"Processed{total} files")


            except Exception as e:
                print(f"Error Processing {file}:{e}")                

    print(f"\n Done! Extracted features from {total} speech files.")
    print(f"Skipped {skipped} song fiels")

    df=pd.DataFrame(rows)
    return df

def show_summary(df):
    print("\n" + "=" * 55)
    print("Dataset Summary")
    print("="*55)
    print(f"Tota Samples : {len(df)}")
    print(f"Features : {N_MFCC} MFCC coeffiecients" )
    print(f"Unique Emotions:{df['emotion'].nunique()}")
    print(f"Unique actors : {df['actor_id'].nunique()}")
    print("\n Samples per emotion:")
    for emo,count in df["emotion"].value_counts().items():
        bar="█" * (count//5)
        print(f"{emo:<12} {bar} ({count})")

def visualize_mfcc(df):
    
    # MFCC heatmap

    emotions=df['emotion'].unique()
    fig,axes=plt.subplots(2,4,figsize=(16,6))
    axes=axes.flatten()

    for i ,emotion in enumerate(sorted(emotions)):
        sample_file=df[df['emotion']==emotion].iloc[0]["file"]

        for root, dirs, files in os.walk(dataset_path):
            if sample_file in files:
                file_path=os.path.join(root,sample_file)
                break

        signal,sr=librosa.load(file_path,sr=sample_rate)
        signal,_=librosa.effects.trim(signal,top_db=20)
        mfccs=librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=N_MFCC)
            
        axes[i].imshow(mfccs,aspect='auto',origin='lower',cmap='viridis')
        axes[i].set_title(f"{emotion}",fontsize=12,fontweight="bold")
        axes[i].set_xlabel("Time Frames")
        axes[i].set_ylabel("MFCC Coefficients")

    plt.suptitle("MFCC Features per Emotion (Ravdess) ",fontsize=14)
    plt.tight_layout()
    plt.savefig("mfcc_visualization.png",dpi=500)
    plt.show()
    print("\n Saved mfcc_visualization.png")

def main():
    df=build_dataset()
    show_summary(df)

    df.to_csv(csv_output,index=False)
    print(f"\n Saved features -> {csv_output}")
    print(f"Shape: {df.shape[0]} rows * {df.shape[1]} cols")

    print("\n Generating MFcc visualization") # MFCC Heatmaps
    visualize_mfcc(df)

    print("\n All Done ! Files created ")
    
if __name__=="__main__":
    main()




