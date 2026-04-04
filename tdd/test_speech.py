from speech_module import predict_emotion, identify_speaker, classify_speech_type

def test_emotion():
    assert predict_emotion(0.8) == "Happy"

def test_speaker():
    assert identify_speaker("audio.wav") == "Sukhman"

def test_speech_type():
    assert classify_speech_type("I will teach today") == "teaching"

emotion = predict_emotion(0.8)
speaker = identify_speaker("audio.wav")
speech_type = classify_speech_type("I will teach today")

print(f"{speaker} is speaking in a {emotion.lower()} tone while {speech_type}.")