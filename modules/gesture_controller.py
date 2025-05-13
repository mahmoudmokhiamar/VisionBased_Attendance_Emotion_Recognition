import cv2
import mediapipe as mp
import math
import numpy as np
import platform
from typing import Optional

class GestureController:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Volume control setup
        self.volume_control = self._init_volume_control()
        
        # Volume UI state
        self.vol_bar = 400
        self.vol_per = 0
        self.bar_x = 50
        self.bar_y_top = 150
        self.bar_y_bottom = 400

    def _init_volume_control(self):
        """Initialize platform-appropriate volume control"""
        system = platform.system()
        if system == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                vol_range = volume.GetVolumeRange()
                return {
                    'volume': volume,
                    'min_vol': vol_range[0],
                    'max_vol': vol_range[1],
                    'available': True
                }
            except ImportError:
                print("pycaw not available - volume control disabled")
        elif system == "Darwin":  # macOS
            print("Volume control not implemented for macOS")
        elif system == "Linux":
            print("Volume control not implemented for Linux")
        
        return {'available': False}

    def set_volume(self, percentage: float):
        """Set system volume if available"""
        if not self.volume_control.get('available', False):
            return
        
        if platform.system() == "Windows":
            vol = np.interp(percentage, [0, 100], 
                          [self.volume_control['min_vol'], self.volume_control['max_vol']])
            self.volume_control['volume'].SetMasterVolumeLevel(vol, None)

    def process_frame(self, frame):
        """Process hand gestures and return frame with visual feedback"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                
                # Get landmark positions
                landmarks = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w = frame.shape[:2]
                    landmarks.append((id, int(lm.x * w), int(lm.y * h)))
                
                if len(landmarks) >= 9:  # Ensure we have thumb and index finger
                    # Thumb (4) and index (8) positions
                    x1, y1 = landmarks[4][1], landmarks[4][2]
                    x2, y2 = landmarks[8][1], landmarks[8][2]
                    
                    # Calculate distance and volume percentage
                    length = math.hypot(x2 - x1, y2 - y1)
                    self.vol_per = np.interp(length, [50, 220], [0, 100])
                    self.vol_bar = np.interp(length, [50, 220], [400, 150])
                    
                    # Set system volume if available
                    self.set_volume(self.vol_per)
                    
                    # Draw volume control UI
                    cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw volume bar (works even without volume control)
        self._draw_volume_bar(frame)
        return frame

    def _draw_volume_bar(self, frame):
        """Draw the volume level bar on the frame"""
        bar_width = 40
        bar_radius = 20
        
        # Background
        cv2.rectangle(frame, (self.bar_x, self.bar_y_top), 
                     (self.bar_x + bar_width, self.bar_y_bottom), 
                     (220, 220, 220), -1)
        
        # Filled volume
        cv2.rectangle(frame, (self.bar_x, int(self.vol_bar)), 
                     (self.bar_x + bar_width, self.bar_y_bottom), 
                     (0, 122, 255), -1)
        
        # Rounded corners
        cv2.circle(frame, (self.bar_x + bar_width // 2, int(self.vol_bar)), 
                  bar_radius, (0, 122, 255), -1)
        cv2.circle(frame, (self.bar_x + bar_width // 2, self.bar_y_bottom), 
                  bar_radius, (0, 122, 255), -1)
        
        # Percentage text
        cv2.putText(frame, f'{int(self.vol_per)}%', 
                   (self.bar_x - 5, self.bar_y_top - 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)
        
        # System availability indicator
        if not self.volume_control.get('available', False):
            cv2.putText(frame, "Volume Control: Simulation", 
                       (self.bar_x, self.bar_y_bottom + 30), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)