import cv2
import time
import os
import numpy as np
from modules.ui_display import UIDisplay
from concurrent.futures import ThreadPoolExecutor
from modules.face_recognition import FaceRecognizer
from modules.emotion_detection import EmotionDetector
from modules.attendance_logger import AttendanceLogger
from modules.gesture_controller import GestureController
from modules.camera_calibration import calibrate_camera, undistort_image

class AttendanceSystem:
    def __init__(self):
        # Configuration
        self.cam_index = 0
        self.detect_interval = 20
        self.infer_size = (480, 360)
        
        # Initialize modules
        self.face_recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()
        self.attendance_logger = AttendanceLogger()
        self.ui = UIDisplay()
        
        # Camera calibration
        self.mtx, self.dist = None, None
        # self._initialize_calibration()
        
        # Camera
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
            
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.last_result = (None, None, False)
        self.frame_count = 0  # Initialize frame counter
        self.gesture_controller = GestureController()
        self.gesture_mode = False  # Toggle with 'g' key
        

    def _initialize_calibration(self):
        """Initialize camera calibration with better error handling"""
        calib_dir = "data/calibration"
        
        # Create directory if it doesn't exist
        os.makedirs(calib_dir, exist_ok=True)
        
        # Check for valid calibration images
        calib_images = [f for f in os.listdir(calib_dir) 
                    if f.lower().endswith(('.jpg', '.png'))]
        
        if not calib_images:
            print(f"No calibration images found in {calib_dir}")
            print("Please add chessboard pattern images and restart")
            self.mtx = self.dist = None
            return
        
        try:
            self.mtx, self.dist = calibrate_camera(calib_images_dir=calib_dir)
            print(f"Successfully calibrated using {len(calib_images)} images")
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            print("Please check your calibration images:")
            print("- Must show a complete 9x6 chessboard pattern")
            print("- Taken from different angles")
            print("- Well-lit and in focus")
            self.mtx = self.dist = None
            
            try:
                self.mtx, self.dist = calibrate_camera(calib_images_dir="data/calibration")
                print("Camera calibration loaded successfully")
            except Exception as e:
                print(f"Calibration skipped: {str(e)}")
                print("Please add chessboard images to data/calibration/ and restart")
                self.mtx = self.dist = None

    def _undistort_frame(self, frame):
        """Apply undistortion if calibration data exists"""
        if self.mtx is not None and self.dist is not None:
            return undistort_image(frame, self.mtx, self.dist)
        return frame

    def _async_process(self, frame):
        """Background processing task"""
        name, confidence = self.face_recognizer.recognize_face(frame)
        emotion = self.emotion_detector.detect(frame)['dominant']
        new_log = False
        
        if name and confidence > 0.6:
            new_log = self.attendance_logger.log(name, emotion)
        
        return name, emotion, new_log

    def run(self):
        try:
            self.pTime = time.time()  # Initialize previous time
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Calculate FPS (works in both modes)
                self.cTime = time.time()
                fps = 1 / (self.cTime - self.pTime)
                self.pTime = self.cTime
                
                # Check for mode toggle key
                key = cv2.waitKey(1)
                if key == ord('g'):
                    self.gesture_mode = not self.gesture_mode
                    print(f"Gesture mode {'enabled' if self.gesture_mode else 'disabled'}")
                
                if self.gesture_mode:
                    # Process gesture control
                    frame = self.gesture_controller.process_frame(frame)
                    
                    # Display mode indicator
                    cv2.putText(frame, "GESTURE CONTROL MODE", (180, 40), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, "Press 'g' to return to attendance mode", (180, 70), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)
                else:
                    # Normal attendance system processing
                    frame = self._undistort_frame(frame)
                    self.frame_count += 1
                    
                    # Schedule inference
                    if self.frame_count % self.detect_interval == 0 and self.future is None:
                        small_frame = cv2.resize(frame, self.infer_size)
                        self.future = self.executor.submit(self._async_process, small_frame)
                    
                    # Get results if available
                    if self.future and self.future.done():
                        self.last_result = self.future.result()
                        self.future = None
                    
                    # Display attendance information
                    name, emotion, new_log = self.last_result
                    frame = self.ui.draw_overlay(frame, name, emotion, new_log)
                    
                    # Show mode indicator
                    cv2.putText(frame, "Press 'g' for gesture control", (10, 20), 
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)
                
                # Display FPS
                cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1]-100, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 180, 20), 2)
                
                # Show the frame
                cv2.imshow("Smart Attendance System", frame)
                
                # Exit on ESC
                if key == 27:
                    break
                    
        finally:
            self.cap.release()
            self.executor.shutdown()
            cv2.destroyAllWindows()
   
if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()