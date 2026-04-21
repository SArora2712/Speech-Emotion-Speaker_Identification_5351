# prepare_wav2vec_data.py
import pandas as pd
import os

print("🔄 Preparing dataset for Wav2Vec2...")

df = pd.read_csv("data/final_combined_dataset.csv", dtype={'speaker': str})

def get_full_path(row):
    if row['source'] == 'ravdess':
        try:
            speaker_id = int(row['speaker'])  # convert to int
            actor_folder = f"Actor_{speaker_id:02d}"
        except:
            actor_folder = row['speaker']  # fallback if something weird

        return os.path.join(
            "data",
            "Ravdess",
            "Audio_Speech_Actors_01-24",
            actor_folder,
            row['file']
        )
    else:
        return os.path.join(
            "data",
            "Custom_Dataset",
            row['speaker'],
            row['emotion'],
            row['file']
        )
df['file_path'] = df.apply(get_full_path, axis=1)

# Validation
df['file_exists'] = df['file_path'].apply(os.path.exists)
print(f"\n✅ Valid audio files found: {df['file_exists'].sum()}/{len(df)}")
print(f"   → RAVDESS: {df[df['source']=='ravdess']['file_exists'].sum()}/1440")
print(f"   → Custom : {df[df['source']=='custom']['file_exists'].sum()}/639")

if df['file_exists'].sum() < len(df):
    print("⚠️  Some files are still missing. Check paths below.")

# Save
df.to_csv("data/wav2vec_ready_dataset.csv", index=False)
print("\n✅ Successfully created: data/wav2vec_ready_dataset.csv")
print(f"Total samples ready: {len(df)}")