import cv2
import numpy as np
from typing import Optional

class UIDisplay:
    def __init__(self):
        self.colors = {
            'happy': (0, 255, 0),
            'neutral': (255, 255, 0),
            'sad': (0, 0, 255),
            'angry': (0, 0, 180),
            'fear': (255, 0, 255),
            'surprise': (0, 255, 255),
            'confused': (255, 0, 0),
            'default': (255, 255, 255)
        }

    def draw_overlay(self, frame: np.ndarray, name: Optional[str], 
                    emotion: str, new_log: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 70), (50, 50, 50), -1)
        
        # Draw face bounding box if recognized
        if name:
            color = self.colors.get(emotion.lower(), self.colors['default'])
            cv2.rectangle(frame, (0, 70), (w, h), color, 2)
            
            # Display info
            cv2.putText(frame, f"User: {name}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if new_log:
                cv2.putText(frame, "RECORDED", (w-120, 45), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.drawMarker(frame, (w-150, 35), (0, 255, 0), 
                             markerType=cv2.MARKER_TILTED_CROSS, thickness=2)
        else:
            cv2.putText(frame, "No authorized face detected", (10, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame