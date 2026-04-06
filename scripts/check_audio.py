import os
import numpy as np
import librosa
import soundfile as sf

# ─────────────────────────────────────────
DATASET_PATH  = "data\Custom_Dataset"
SAMPLE_RATE   = 22050
MIN_DURATION  = 1.0    # seconds — too short means person did not speak
MAX_DURATION  = 6.0    # seconds — too long means too much silence
MIN_ENERGY    = 0.001  # too quiet means background noise or no speech
SILENCE_RATIO = 0.60   # if more than 60% is silence something is wrong
# ─────────────────────────────────────────

def check_audio(file_path):
    """
    Check one audio file and return a list of problems found.
    Returns empty list if file is good.
    """
    problems = []

    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        return [f"Cannot load file: {e}"]

    # Check 1 — duration
    duration = len(signal) / sr
    if duration < MIN_DURATION:
        problems.append(f"Too short ({duration:.1f}s) — speaker may not have spoken")
    if duration > MAX_DURATION:
        problems.append(f"Too long ({duration:.1f}s) — too much silence")

    # Check 2 — energy (volume)
    energy = np.sqrt(np.mean(signal ** 2))
    if energy < MIN_ENERGY:
        problems.append(f"Too quiet (energy={energy:.4f}) — possible background noise only")

    # Check 3 — silence ratio
    trimmed, _ = librosa.effects.trim(signal, top_db=20)
    silence_ratio = 1 - (len(trimmed) / len(signal))
    if silence_ratio > SILENCE_RATIO:
        problems.append(
            f"Too much silence ({silence_ratio*100:.0f}%) — "
            f"speaker paused too long or did not speak clearly"
        )

    # Check 4 — clipping (recording was too loud)
    max_amp = np.max(np.abs(signal))
    if max_amp >= 0.99:
        problems.append(f"Audio clipped (max amplitude={max_amp:.3f}) — recording was too loud")

    # Check 5 — MFCC variation (checks if speech is actually present)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_std = np.mean(np.std(mfccs, axis=1))
    if mfcc_std < 2.0:
        problems.append(
            f"Very low speech variation (std={mfcc_std:.2f}) — "
            f"may be silence or monotone noise"
        )

    return problems


def check_all():
    print("\n" + "=" * 60)
    print("  AUDIO QUALITY CHECK")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"  Folder not found: {DATASET_PATH}")
        return

    total_files   = 0
    good_files    = 0
    bad_files     = 0
    all_problems  = []

    for speaker in sorted(os.listdir(DATASET_PATH)):
        speaker_path = os.path.join(DATASET_PATH, speaker)
        if not os.path.isdir(speaker_path):
            continue

        speaker_bad = 0
        print(f"\n  Speaker: {speaker}")
        print("  " + "-" * 50)

        for emotion in sorted(os.listdir(speaker_path)):
            emo_path = os.path.join(speaker_path, emotion)
            if not os.path.isdir(emo_path):
                continue

            files = sorted([f for f in os.listdir(emo_path)
                            if f.endswith(".wav")])

            for fname in files:
                fpath    = os.path.join(emo_path, fname)
                problems = check_audio(fpath)
                total_files += 1

                if problems:
                    bad_files   += 1
                    speaker_bad += 1
                    print(f"  BAD  {emotion}/{fname}")
                    for p in problems:
                        print(f"       → {p}")
                    all_problems.append((fpath, problems))
                else:
                    good_files += 1
                    print(f"  OK   {emotion}/{fname}")

        if speaker_bad == 0:
            print(f"  All recordings for {speaker} passed.")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total files checked : {total_files}")
    print(f"  Good files          : {good_files}")
    print(f"  Bad files           : {bad_files}")

    if bad_files == 0:
        print("\n  All recordings are good quality!")
    else:
        pct = (bad_files / total_files) * 100
        print(f"\n  {pct:.0f}% of recordings have issues.")
        print("\n  Files to re-record:")
        for fpath, problems in all_problems:
            print(f"\n  {fpath}")
            for p in problems:
                print(f"    → {p}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_all()