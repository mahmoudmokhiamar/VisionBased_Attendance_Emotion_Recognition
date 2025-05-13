import os
import cv2
import json
import tempfile
import numpy as np
from typing import Optional, Tuple, Set
from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, db_path: str = "data/faces", auth_file: str = "authorized.json", threshold: float = 0.6):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.threshold = threshold
        self.authorized = self._load_authorized(auth_file)
        self.recognized = set()

    def _load_authorized(self, auth_file: str) -> Set[str]:
        if os.path.exists(auth_file):
            with open(auth_file) as f:
                return set(json.load(f))
        return set()

    def register_face(self, img: np.ndarray, user_id: str):
        user_dir = os.path.join(self.db_path, user_id)
        os.makedirs(user_dir, exist_ok=True)
        count = len(os.listdir(user_dir))
        cv2.imwrite(f"{user_dir}/{count+1}.jpg", img)

    def recognize_face(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
                cv2.imwrite(tmp.name, img_rgb)
                results = DeepFace.find(
                    img_path=tmp.name,
                    db_path=self.db_path,
                    model_name="Facenet",
                    distance_metric="cosine",
                    enforce_detection=False,
                    detector_backend="retinaface",
                    align=True,
                    silent=True
                )

                if results and isinstance(results, list) and len(results[0]) > 0:
                    best = results[0].iloc[0]
                    if best['distance'] < self.threshold:
                        user_id = os.path.basename(os.path.dirname(best['identity']))
                        if not self.authorized or user_id in self.authorized:
                            return user_id, 1 - best['distance']
        except Exception as e:
            if "Face could not be detected" not in str(e):
                print(f"[FACE] Error: {str(e)}")
        return None, 0.0