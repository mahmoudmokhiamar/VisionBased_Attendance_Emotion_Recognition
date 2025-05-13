import cv2
from deepface import DeepFace
import os
import tempfile

def identify_face(frame, faces_dir="data/faces"):
    """
    Identify a person from the frame using DeepFace and images in faces_dir.
    """
    try:
        # Save the current frame as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, frame)  # Save the frame as a temporary image file

            # Perform face recognition using DeepFace
            result = DeepFace.find(img_path=temp_file.name, db_path=faces_dir, enforce_detection=False)

            # Check if results are found
            if result and len(result[0]) > 0:
                # Extract the identity of the matched face (the person's folder name)
                identity_path = result[0].iloc[0]['identity']
                person_name = os.path.basename(os.path.dirname(identity_path))  # Get the folder name as person's name
                return person_name
            else:
                print("[ERROR] No matching face found.")
    except Exception as e:
        print(f"[ERROR] Face recognition failed: {e}")
    return None
