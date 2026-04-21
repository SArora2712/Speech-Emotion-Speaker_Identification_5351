import os

dataset_path="data\Ravdess"

total_files=0
speech_count=0
song_count=0

for root,dirs,files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            total_files+=1
            if "speech" in root.lower():
                speech_count+=1
            elif "song" in root.lower():
                song_count+=1

print("Total .wav files: ",total_files)
print("Speech Files: ",speech_count)
print("Song Files: ",song_count)