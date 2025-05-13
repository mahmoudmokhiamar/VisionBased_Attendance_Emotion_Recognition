import os
import pandas as pd
from typing import Optional
from datetime import datetime

class AttendanceLogger:
    def __init__(self, file_path: str = "attendance.xlsx"):
        self.file_path = file_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M")
        self.logged = set()
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.file_path):
            pd.DataFrame(columns=['Name', 'Emotion', 'Timestamp', 'Session']).to_excel(self.file_path, index=False)

    def log(self, name: str, emotion: str) -> bool:
        if name in self.logged:
            return False
            
        record = {
            'Name': name,
            'Emotion': emotion,
            'Timestamp': datetime.now(),
            'Session': self.session_id
        }

        try:
            if self.file_path.endswith('.xlsx'):
                df = pd.read_excel(self.file_path)
                df = pd.concat([df, pd.DataFrame([record])])
                df.to_excel(self.file_path, index=False)
            else:
                df = pd.DataFrame([record])
                df.to_csv(self.file_path, mode='a', header=not os.path.exists(self.file_path), index=False)
            
            self.logged.add(name)
            return True
        except Exception as e:
            print(f"[ATTENDANCE] Error: {str(e)}")
            return False