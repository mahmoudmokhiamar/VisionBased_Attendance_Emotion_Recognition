import cv2
from modules.face_recognition import identify_face
from modules.emotion_detection import detect_emotion
from modules.attendance_logger import log_attendance
from modules.camera_calibration import calibrate_camera
from modules.ui_display import draw_overlay

def main():
    # Attempt camera calibration
    try:
        camera_matrix, dist_coeffs = calibrate_camera()
        print("[INFO] Camera calibrated.")
    except Exception as e:
        print(f"[WARNING] Calibration skipped: {e}")
        camera_matrix = None
        dist_coeffs = None

    cap = cv2.VideoCapture(0)
    recognized = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort if calibration available
        if camera_matrix is not None and dist_coeffs is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Detect face and emotion
        name = identify_face(frame)
        emotion = detect_emotion(frame)
        attendance_marked = False

        if name and name not in recognized:
            log_attendance(name, emotion)
            recognized.add(name)
            attendance_marked = True

        # Draw overlay (name and emotion) on the live camera frame
        frame = draw_overlay(frame, name, emotion, attendance_marked)

        # Display the frame with the overlay
        cv2.imshow("Vision Attendance & Emotion", frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
