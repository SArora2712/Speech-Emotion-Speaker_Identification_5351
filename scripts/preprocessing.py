import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# In BOTH preprocessing.py and feature_extraction.py, line 9:
DATASET_PATH = "data\Ravdess"
sample_rate=22050

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
def get_emotion_from_filename(file_name):
    """
    RAVDESS filename: 03-01-05-01-02-01-12.wav
    Index 2 = emotion code
    """
    try:
        parts = file_name.replace(".wav", "").split("-")
        return emotion_map.get(parts[2], "unknown")
    except:
        return "unknown"
    
def Load_Preprocess(file_path):

    #Load audio at standard sample rate
    signal,sr=librosa.load(file_path,sr=sample_rate)

    #Normalize amplitude to range [-1,1]
    if np.max(np.abs(signal))>0:
        signal=signal/np.max(np.abs(signal))

    #Trim leading and trailing silence
    signal_trimmed,trim_index=librosa.effects.trim(signal,top_db=20)

    return signal, signal_trimmed,sr,trim_index

def preprocessing_steps_visualization(file_path,emotion_label):

    signal_raw,signal_clean,sr,trim_idx=Load_Preprocess(file_path)

    fig,axes=plt.subplots(3,1,figsize=(12,8))

    # Raw Waveform
    axes[0].plot(signal_raw,color="#4A90D9",linewidth=0.5)
    axes[0].set_title(f" Raw Audio - emotions: {emotion_label}",fontweight="bold")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Amplitude")

    #Normalized Waveform
    signal_norm=signal_raw/np.max(np.abs(signal_raw))
    axes[1].plot(signal_norm,color="#27AE60",linewidth=0.5)
    axes[1].set_title("Normalized Auio",fontweight="bold")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Amplitude")

    #Trimmed Waveform
    axes[2].plot(signal_clean,color="#E74C3C", linewidth=0.5)
    trim_start=trim_idx[0]
    trim_end=trim_idx[1]
    axes[2].set_title("Silence trimmed",fontweight="bold")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Amplitude")

    plt.suptitle("Audio Preprocessing Pipeline (Ravdess) ",fontsize=14,y=1.01)
    plt.tight_layout()
    plt.savefig("peprocessing_steps.png",dpi=150,bbox_inches="tight")
    plt.show()
    print("Saved! preprocessing_steps.png")


def Spectogram_plot(file_path,emotion_label):
    """
    A mel spectogram generated whic visually displays frequency patterns over time.

    """
    _,signal_clean,sr,_=Load_Preprocess(file_path)

    mel_spec=librosa.feature.melspectrogram(y=signal_clean,sr=sr,n_mels=128)
    mel_db=librosa.power_to_db(mel_spec,ref=np.max)

    plt.figure(figsize=(10,4))
    librosa.display.specshow(mel_db,sr=sr,x_axis="time",y_axis="mel",cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram-Emotion:{emotion_label}",fontweight="bold")
    plt.tight_layout()
    plt.savefig("spectogram.png",dpi=50)
    plt.show()
    print("Saved! spectogram.png")


def sample_demo():
    print("\n"+"="*55)
    print("Audio Preprocessing Demo")
    print("="*55)

    emotion_files={}

    speech_files = []
 
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith(".wav"):
                continue
            if "speech" in root.lower():
                full_path = os.path.join(root, file)
                speech_files.append((full_path, file))
 
      

    emotion_files = {}
    for full_path, file_name in speech_files:
        emotion = get_emotion_from_filename(file_name)
        if emotion not in emotion_files:
            emotion_files[emotion] = full_path
 
    print(f"\n {'Emotion':<12} {'Raw samples':>12} {'After trim':>12} {'Reduction':>10}")
    print(" "+"-"*50)

    first_file=None
    first_label=None

    for emotion in sorted(emotion_files.keys()):
        file_path=emotion_files[emotion]
        raw, clean, sr,_=Load_Preprocess(file_path)
        reduction=(1-len(clean)/len(raw))*100
        print(f"{emotion:<12} {len(raw):>12,} {len(clean):>12} {reduction:>9.1f}%")

        if first_file is None:
            first_file=file_path
            first_label=emotion

    print(f"\n Generating plots for sample:{first_label}..")            
    preprocessing_steps_visualization(first_file,first_label)
    Spectogram_plot(first_file,first_label)

    print("\n Preprocessing demo complete!")
    print("Files Saved:")
    print("preprocessing_steps.png")
    print('spectogram.png')

if __name__=="__main__":
    sample_demo()






