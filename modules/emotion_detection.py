import cv2
import numpy as np
from typing import Dict
from collections import deque
from deepface import DeepFace

class EmotionDetector:
    def __init__(self, threshold: float = 0.4, stability_window: int = 2):
        """
        Args:
            threshold: Confidence threshold for emotion detection
            stability_window: Number of frames to consider for stable emotion
        """
        self.emotions = ['happy', 'sad', 'neutral', 'angry', 'fear', 'surprise', 'confused']
        self.threshold = threshold
        self.stability_window = stability_window
        self.emotion_history = deque(maxlen=stability_window)
        self.current_stable_emotion = 'neutral'

    def detect(self, img: np.ndarray) -> Dict:
        """
        Returns dictionary with 'dominant' emotion key
        (maintains compatibility with existing main.py)
        """
        try:
            # Convert color space if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            # Perform emotion analysis
            result = DeepFace.analyze(
                img_path=img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                scores = result[0]['emotion']
                current_emo = result[0]['dominant_emotion']
                
                # Special handling for confused state
                if (scores['fear'] > self.threshold and 
                    scores['surprise'] > self.threshold):
                    current_emo = 'confused'
                
                # Update emotion history
                self.emotion_history.append(current_emo)
                
                # Determine most frequent emotion in history
                if len(self.emotion_history) == self.stability_window:
                    counts = {}
                    for emo in self.emotion_history:
                        counts[emo] = counts.get(emo, 0) + 1
                    self.current_stable_emotion = max(counts.items(), key=lambda x: x[1])[0]
                
                return {'dominant': self.current_stable_emotion}
                
        except Exception as e:
            print(f"[EMOTION] Error: {str(e)}")
        
        return {'dominant': self.current_stable_emotion}