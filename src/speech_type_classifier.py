import speech_recognition as sr
import librosa
import soundfile as sf
import numpy as np
import os
import tempfile


Speech_type_keywords={
    "teaching":{
        "keywords":[
            "step", "first", "second", "third", "fourth", "fifth",
            "listen", "repeat", "remember", "note", "write",
            "understand", "explain", "lesson", "learn", "class",
            "formula", "definition", "example", "concept",
            "attention", "important", "exam", "question", "answer",
            "correct", "wrong", "try", "practice", "homework",
            "today we", "let me show", "follow", "instructions",
            "now do", "next step", "open", "click", "type", "press",
            "look at", "observe", "notice", "see how", "watch"
        ],
        "phrases":[ "pay attention",
            "step one",
            "step two",
            "step three",
            "let me explain",
            "write this down",
            "this is important",
            "today we will",
            "now repeat",
            "the answer is",
            "do you understand",
            "listen carefully",
            "follow these steps",
            "note this down",
            ],
            "weight":1.0

    },
    "knowledge_sharing": {
        "keywords": [
            "basically", "actually", "essentially", "fundamentally",
            "works", "means", "defined", "called", "known",
            "reason", "because", "therefore", "hence", "thus",
            "research", "study", "data", "process", "system",
            "method", "technique", "algorithm", "model", "feature",
            "according", "found", "shows", "indicates", "suggests",
            "interesting", "fascinating", "fact", "information",
            "technology", "science", "machine", "learning", "network",
            "signal", "frequency", "coefficient", "extract", "train",
            "dataset", "accuracy", "predict", "classify", "detect"
        ],
        "phrases": [
            "basically what happens",
            "the reason for this",
            "this is because",
            "works by",
            "is defined as",
            "is known as",
            "according to",
            "research shows",
            "the idea is",
            "what this means",
            "in simple terms",
            "to put it simply",
            "the concept is",
            "this process",
            "the system",
        ],
        "weight": 1.0
    },
 
    "storytelling": {
        "keywords": [
            "happened", "remember", "suddenly", "eventually",
            "finally", "then", "after", "before", "during",
            "once", "day", "time", "night", "moment",
            "felt", "thought", "realized", "decided", "started",
            "went", "came", "saw", "heard", "told",
            "friend", "family", "trip", "journey", "experience",
            "never", "always", "used to", "back then", "ago",
            "story", "incident", "funny", "scary", "strange"
        ],
        "phrases": [
            "so what happened",
            "i remember when",
            "this one time",
            "one day",
            "it was",
            "we were",
            "i was",
            "and then",
            "after that",
            "the next day",
            "believe it or not",
            "long story short",
            "out of nowhere",
            "all of a sudden",
            "you will not believe",
        ],
        "weight": 1.0
    },
     "conversation": {
        "keywords": [
            "bro", "yaar", "dude", "man", "buddy",
            "right", "okay", "ok", "yeah", "yep",
            "honestly", "seriously", "literally", "actually",
            "anyway", "whatever", "totally", "definitely",
            "cool", "nice", "awesome", "great", "wow",
            "did you", "have you", "are you", "do you",
            "what do", "how are", "where are", "when did",
            "let us", "come on", "no way", "by the way",
            "guess what", "you know", "i mean", "kind of",
            "sort of", "like", "think", "feel", "know"
        ],
        "phrases": [
            "did you see",
            "have you heard",
            "by the way",
            "guess what",
            "no way",
            "you know what",
            "i was thinking",
            "what do you think",
            "how are you",
            "what happened",
            "tell me",
            "come on",
            "let us go",
            "are you serious",
            "that is crazy",
        ],
        "weight": 0.8  
    }

}

speech_type_desc={
    "teaching":"while teaching",
    "knowledge_sharing":"while sharing knowledge",
    "storytelling":"while narrating a story",
    "conversation":"while having a conversation"
}

def convert_audio_to_wav(file_path):
    #Check if audio is in .wav format ,if not then converting it into .wav format

    if file_path.endswith(".wav"):
        return file_path,False
    
    signal,sr=librosa.loud(file_path,sr=16000,mono=True)
    temp_file=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
    sf.write(temp_file.name,signal,sr)
    return temp_file.name,True

def speech_to_text(file_path):
    
    ## COnverting audio file to the text file using Speechrecognition

    recognizer=sr.Recognizer()
    wav_path,is_temp=convert_audio_to_wav(file_path)

    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source,duration=0.3)
            audio=recognizer.record(source)

        text=recognizer.recognize_google(audio)
        print(f"Transcript:\"{text}\"")
        return text.lower()
    
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition service error:{e}")
        print("Check internet connection ")
        return " "
    finally:
        if is_temp and os.path.exists(wav_path):
            os.remove(wav_path)

def calculate_score(text):
    #Calculatinf Keyword match score for each sppech type PhraseScore-> 2 points  KeywordsScore-> 1 points          

    scores={speech_type:0.0 for speech_type in Speech_type_keywords}

    if not text:
        return scores
    
    for speech_type,data in Speech_type_keywords.items():
        score=0
        weight=data['weight']

        #Check Keywords 
        for keyword in data["keywords"]:
            if keyword in text:
                score+=1

        #Check Phrases
        for phrase in data['phrases']:
            if phrase in text:
                score+=2
        
        scores[speech_type]=score*weight

    return scores

def classify_speech_type(file_path) :
    print(f'\n Classifying speech type for:{os.path.basename(file_path)}')

    #1->converting speech to text
    transcript=speech_to_text(file_path)

    #2->Calculating keyword and phrases scores
    scores=calculate_score(transcript)

    #3->finding highest score category
    if all(s==0 for s in scores.values()):
        speech_type="conversation"
        confidence="low"
    else:
        speech_type=max(scores,key=scores.get)
        top_score=scores[speech_type]

        sorted_scores=sorted(scores.values(),reverse=True)
        gap=sorted_scores[0]-sorted_scores[1] if len(sorted_scores) > 1 else top_score

        if top_score>=5 and gap>=3:
            confidence="high"
        elif top_score>=2:
            confidence="medium"
        else:
            confidence="low"

    return speech_type,transcript,scores,confidence

def generate_output(speaker,emotion,speech_type):
    desc=speech_type_desc.get(speech_type,"during a conversation")
    return f"{speaker} is speaking in a {emotion} tone {desc}."

def analyze_file(file_path,speaker='Speaker',emotion="neutral"):
    print("\n"+'-'*40)
    print("\t SPEECH TYPE ANALYSIS")
    print('-'*40)
    print(f"File:{os.path.basename(file_path)}")
    print(f"Speaker: {speaker}")
    print(f"Emotion: {emotion}")

    speech_type,transcript,scores,confidence=classify_speech_type(file_path)

    print(f"\n Scores per category  ")
    for st,score in sorted(scores.items(),key=lambda x:x[1],reverse=True):
        bar="█"*int(score) if score >0 else ""
        marker="<--selected" if st==speech_type else ""
        print(f"{st:<20}{bar}({score:.1f}){marker}")

    print(f"\n Speech type: {speech_type}")
    print(f"Confience:{confidence}")

    sentence=generate_output(speaker,emotion,speech_type)
    print(f"\n OUtput:\"{sentence}\"")
    print("-"*50)

    return speech_type,sentence

def demo():
    print("\n"+"-"*50)
    print("SPEECH TYPE CLASSIFIER (DEMO)")
    print('-'*50)
    file_path=input("\n EN path to a .wav file to test:").strip()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    speaker=input("Enter speaker name :").strip()
    emotion=input("ENtr emotion: ").strip()

    speaker=speaker if speaker else "Speaker"
    emotion=emotion if emotion else "Neutral"

    speech_type,sentence=analyze_file(file_path,speaker,emotion)

    print(f"\n Final Output sentence:")
    print(f" ---> {sentence}")

def test_text(text,speaker="Sukhman",emotion='happy'):
    print("\n"+"-"*50)
    print("Text-Based Test")
    print("-"*50)
    print(f"Input text: \"{text}\"")

    scores=calculate_score(text.lower())
    speech_type=max(scores,key=scores.get) if any(
        s>0 for s in scores.values()
    ) else "conversation"

    print(f"\n Scores:")
    for st,score in sorted(scores.items(),key=lambda x:x[1],reverse=True):
        bar="█"*int(score) if score>0 else ""
        marker="<---selected" if st==speech_type else ""
        print(f"{st:<20} {bar} ({score:.1f}) {marker}")

    sentence=generate_output(speaker,emotion,speech_type)
    print(f"Output: \"{sentence}\"")
    print("-"*50)
    return speech_type

def main():
    while True:
        print("\n"+"-"*50)
        print("SPEECH TYPE CLASSIFIER")
        print("-"*50)
        print("1. Analyze an audio file")
        print("2. Test with typed text")
        print("3. Exit")

        choice=input("\n Enter Choice: ").strip()
        if choice=="1":
            demo()

        elif choice=='2':
            print("\n Type a sentence as if you were speaking: ")
            text=input("\n Your text: ").strip()
            speaker=input("Speaker name: ").strip() or "Speaker"
            emotion=input("Emotion: ").strip() or "neutral"
            test_text(text,speaker,emotion)

        elif choice=='3':
            print("\n Goodbye")
            break

        else:
            print("Invalid Choice")


if __name__=="__main__":
    main()



    
    












