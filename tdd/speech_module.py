def predict_emotion(score):
    return "Happy" if score > 0.7 else "Sad" if score < 0.3 else "Neutral"

def identify_speaker(audio_file):
    return "Sukhman"


def classify_speech_type(text):
    if "teach" in text.lower():
        return "teaching"
    elif "story" in text.lower():
        return "storytelling"
    else:
        return "conversation"
    
    