from deepface import DeepFace

def detect_emotion(frame):
    """
    Detect the emotion from a frame using DeepFace.
    Returns one of: happy, sad, neutral, angry, surprised, etc.
    """
    try:
        analysis = DeepFace.analyze(img_path=frame, actions=["emotion"], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print(f"[ERROR] Emotion detection failed: {e}")
        return "unknown"
