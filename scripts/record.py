import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
DURATION      = 4
CHANNELS      = 1
OUTPUT_FOLDER = "Custom_Dataset"
TARGET_TAKES  = 5    # minimum takes needed per emotion per speaker

# ── RAVDESS-aligned emotions only ────────────────────────────
# These match exactly with RAVDESS emotion codes:
# 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# ── Emotion-specific sentences ────────────────────────────────
EMOTION_SENTENCES = {
    "neutral": [
        "I am going to the market today.",
        "The meeting starts at ten in the morning.",
        "Please submit the report by Friday.",
        "The library closes at eight in the evening.",
        "I will call you back in some time.",
    ],
    "calm": [
        "Everything is going to be fine.",
        "Take a deep breath and relax.",
        "We have plenty of time, there is no rush.",
        "I am feeling very peaceful right now.",
        "Sit down and let us talk about this slowly.",
    ],
    "happy": [
        "I just got the best news of my life!",
        "Today has been such a wonderful day!",
        "I am so excited about this, I cannot believe it!",
        "We finally did it, this is amazing!",
        "I love spending time with all of you!",
    ],
    "sad": [
        "I miss the people who are no longer here.",
        "Nothing feels right anymore, I am very tired.",
        "I cannot believe this happened to me.",
        "I feel so alone and lost right now.",
        "Everything reminds me of what I lost.",
    ],
    "angry": [
        "I cannot believe you did this again!",
        "This is completely unacceptable behavior!",
        "How many times do I have to say this?",
        "You never listen to what I am saying!",
        "I am done, I have had enough of this!",
    ],
    "fearful": [
        "I do not know what is going to happen next.",
        "Something does not feel right, I am scared.",
        "Please do not leave me alone here.",
        "I heard a strange noise, I am very nervous.",
        "I am not sure I can handle this situation.",
    ],
    "disgust": [
        "I cannot even look at this, it is horrible.",
        "This is absolutely disgusting, take it away.",
        "How could anyone do something like this?",
        "I feel sick just thinking about it.",
        "This is the worst thing I have ever seen.",
    ],
    "surprised": [
        "Oh my goodness, I did not expect this at all!",
        "Wait, are you serious? I cannot believe it!",
        "What? How did this even happen?",
        "I am completely shocked right now!",
        "No way! This is unbelievable!",
    ],
}

# ── Acting tips per emotion ───────────────────────────────────
ACTING_TIPS = {
    "neutral":   "Speak normally, no emotion, like reading an announcement.",
    "calm":      "Speak slowly and softly, very relaxed and peaceful.",
    "happy":     "Smile while speaking, raise pitch slightly, speak with energy.",
    "sad":       "Speak slowly, drop your voice, sound tired and heavy.",
    "angry":     "Speak louder, stress words hard, sound very frustrated.",
    "fearful":   "Speak hesitantly, add small pauses, sound nervous.",
    "disgust":   "Speak with a tone of strong disapproval and revulsion.",
    "surprised": "Widen your eyes, raise pitch sharply, sound shocked.",
}

# ── Known speakers ────────────────────────────────────────────
KNOWN_SPEAKERS = [
    "Sukhman"
]
# ─────────────────────────────────────────────────────────────


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def get_file_count(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".wav")])


def get_next_index(folder):
    existing = [f for f in os.listdir(folder) if f.endswith(".wav")]
    return len(existing) + 1


def get_speaker_status(speaker):
    status = {}
    speaker_path = os.path.join(OUTPUT_FOLDER, speaker)
    for emotion in EMOTIONS:
        emo_path      = os.path.join(speaker_path, emotion)
        status[emotion] = get_file_count(emo_path)
    return status


def show_speaker_status(speaker):
    status     = get_speaker_status(speaker)
    incomplete = []
    complete   = []

    for emotion, count in status.items():
        if count >= TARGET_TAKES:
            complete.append((emotion, count))
        else:
            incomplete.append((emotion, count))

    print(f"\n  Status for {speaker}  (target: {TARGET_TAKES} per emotion)")
    print("  " + "-" * 50)

    if incomplete:
        print("  Still needed:")
        for emotion, count in incomplete:
            needed  = TARGET_TAKES - count
            filled  = "█" * count
            empty   = "░" * (TARGET_TAKES - count)
            print(f"    {emotion:<12} {filled}{empty}  {count}/{TARGET_TAKES}  (+{needed})")

    if complete:
        print("\n  Complete:")
        for emotion, count in complete:
            bar = "█" * TARGET_TAKES
            print(f"    {emotion:<12} {bar}  {count}/{TARGET_TAKES}  done")

    total        = sum(status.values())
    total_needed = len(EMOTIONS) * TARGET_TAKES
    print(f"\n  Total: {total}/{total_needed}")
    print("  " + "-" * 50)
    return incomplete


def show_all_speakers_summary():
    print("\n" + "=" * 55)
    print("  DATASET OVERVIEW")
    print("=" * 55)

    if not os.path.exists(OUTPUT_FOLDER):
        print("  No recordings found yet.")
        return

    all_speakers = set(KNOWN_SPEAKERS)
    if os.path.exists(OUTPUT_FOLDER):
        for name in os.listdir(OUTPUT_FOLDER):
            if os.path.isdir(os.path.join(OUTPUT_FOLDER, name)):
                all_speakers.add(name)

    for speaker in sorted(all_speakers):
        speaker_path = os.path.join(OUTPUT_FOLDER, speaker)
        status       = get_speaker_status(speaker)
        total        = sum(status.values())
        target       = len(EMOTIONS) * TARGET_TAKES
        incomplete   = [e for e, c in status.items() if c < TARGET_TAKES]
        pct          = int((total / target) * 100)
        bar          = "█" * (pct // 10) + "░" * (10 - pct // 10)

        if total == 0:
            print(f"  {speaker:<14} [no recordings yet]")
        elif incomplete:
            print(f"  {speaker:<14} {bar} {total}/{target}  needs: {', '.join(incomplete)}")
        else:
            print(f"  {speaker:<14} {bar} {total}/{target}  complete")

    print("=" * 55)


def record_audio():
    print(f"\n  Recording for {DURATION} seconds... Speak now!")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()
    print("  Done.")
    return audio


def record_session():
    print("\n" + "=" * 55)
    print("  CUSTOM DATASET RECORDER")
    print("  All emotions matched to RAVDESS")
    print("=" * 55)

    show_all_speakers_summary()

    # ── Select speaker ────────────────────────────────────────
    all_speakers = list(KNOWN_SPEAKERS)
    existing = []
    if os.path.exists(OUTPUT_FOLDER):
        for name in os.listdir(OUTPUT_FOLDER):
            if os.path.isdir(os.path.join(OUTPUT_FOLDER, name)) and name not in all_speakers:
                existing.append(name)
    all_speakers = sorted(set(all_speakers + existing))

    print("\n  Select speaker:")
    for i, name in enumerate(all_speakers, 1):
        status     = get_speaker_status(name)
        total      = sum(status.values())
        target     = len(EMOTIONS) * TARGET_TAKES
        incomplete = [e for e, c in status.items() if c < TARGET_TAKES]

        if total == 0:
            tag = "  [no recordings yet]"
        elif incomplete:
            tag = f"  [needs: {', '.join(incomplete)}]"
        else:
            tag = "  [complete]"

        print(f"  {i}. {name}{tag}")

    print(f"  {len(all_speakers)+1}. Add new speaker")

    choice = int(input("\n  Select number: "))
    if choice == len(all_speakers) + 1:
        speaker = input("  Enter new speaker name: ").strip()
    else:
        speaker = all_speakers[choice - 1]

    incomplete_emotions = show_speaker_status(speaker)

    # ── Select emotion ────────────────────────────────────────
    print("\n  Select emotion:")
    print("  (RAVDESS codes shown — your model is trained on these same emotions)")

    ravdess_codes = {
        "neutral": "01", "calm": "02", "happy": "03", "sad": "04",
        "angry": "05", "fearful": "06", "disgust": "07", "surprised": "08"
    }

    for i, emotion in enumerate(EMOTIONS, 1):
        speaker_path = os.path.join(OUTPUT_FOLDER, speaker)
        emo_path     = os.path.join(speaker_path, emotion)
        count        = get_file_count(emo_path)
        needed       = max(0, TARGET_TAKES - count)
        code         = ravdess_codes[emotion]

        if needed > 0:
            tag = f"  <- needs {needed} more"
        else:
            tag = "  done"

        print(f"  {i}. [{code}] {emotion:<12} {count}/{TARGET_TAKES}{tag}")

    emo_choice = int(input("\n  Select emotion number: ")) - 1
    emotion    = EMOTIONS[emo_choice]

    # ── Show acting tip ───────────────────────────────────────
    print(f"\n  How to express {emotion.upper()}:")
    print(f"  → {ACTING_TIPS[emotion]}")

    # ── Select sentence ───────────────────────────────────────
    sentences = EMOTION_SENTENCES[emotion]
    print(f"\n  Choose a sentence (these fit the {emotion} emotion):")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    print(f"  {len(sentences)+1}. Type your own")

    sent_choice = int(input("\n  Select sentence: "))
    if sent_choice == len(sentences) + 1:
        sentence = input("  Type sentence: ").strip()
    else:
        sentence = sentences[sent_choice - 1]

    # ── Number of takes ───────────────────────────────────────
    current = get_file_count(os.path.join(OUTPUT_FOLDER, speaker, emotion))
    needed  = max(0, TARGET_TAKES - current)
    takes   = int(input(f"\n  How many takes? (need {needed} more to complete): "))

    # ── Create folder ─────────────────────────────────────────
    save_folder = os.path.join(OUTPUT_FOLDER, speaker, emotion)
    create_folder(save_folder)

    print(f"\n  {'─'*48}")
    print(f"  Speaker  : {speaker}")
    print(f"  Emotion  : {emotion}  (RAVDESS code {ravdess_codes[emotion]})")
    print(f"  Sentence : {sentence}")
    print(f"  Takes    : {takes}")
    print(f"  {'─'*48}")

    # ── Record takes ──────────────────────────────────────────
    for take in range(1, takes + 1):
        print(f"\n  [ Take {take}/{takes} ]")
        print(f"  Acting tip: {ACTING_TIPS[emotion]}")
        print(f"  Say this:  \"{sentence}\"")
        input("\n  Press ENTER when ready...")

        audio = record_audio()

        idx   = get_next_index(save_folder)
        fname = f"{speaker}_{emotion}_{idx:03d}.wav"
        fpath = os.path.join(save_folder, fname)
        sf.write(fpath, audio, SAMPLE_RATE)
        print(f"  Saved -> {fname}")

        replay = input("  Play back? (y/n): ").strip().lower()
        if replay == "y":
            print("  Playing...")
            sd.play(audio, SAMPLE_RATE)
            sd.wait()

        keep = input("  Keep? (y/n): ").strip().lower()
        if keep == "n":
            os.remove(fpath)
            print("  Discarded.")

    print(f"\n  Updated status for {speaker}:")
    show_speaker_status(speaker)


def main():
    while True:
        print("\n" + "=" * 55)
        print("  MAIN MENU")
        print("=" * 55)
        print("  1. Record new audio")
        print("  2. View full dataset summary")
        print("  3. View one speaker in detail")
        print("  4. Exit")

        choice = input("\n  Enter choice: ").strip()

        if choice == "1":
            record_session()

        elif choice == "2":
            show_all_speakers_summary()

        elif choice == "3":
            show_all_speakers_summary()
            name = input("\n  Enter speaker name to view: ").strip()
            show_speaker_status(name)

        elif choice == "4":
            print("\n  Goodbye!")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()