import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
# Path of one .wav file
file_path = r"Ravdess\Audio_Speech_Actors_01-24\Actor_16\03-01-02-01-01-02-16.wav"

# Extracting File name 
file_name=os.path.basename(file_path)

# Load audio
signal, sample_rate = librosa.load(file_path, sr=None)

# Identifying Type of audio by file name
modality_code=file_name.split('-')[1]
if modality_code=='01':
    print("\nSpeech")
if modality_code=='02':
    print("\nSong")  

# Displaying audio features
print("Sample Rate:", sample_rate)
print("Signal Shape:", signal.shape)
print("Data Type:", signal.dtype)
print("Duration (seconds):", len(signal)/sample_rate)

# Plot waveform
plt.figure(figsize=(10,4))
plt.plot(signal)
plt.title("Waveform of Audio File")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()