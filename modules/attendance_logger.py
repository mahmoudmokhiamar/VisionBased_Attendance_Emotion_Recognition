import pandas as pd
from datetime import datetime

def log_attendance(name, emotion, csv_path="attendance.csv"):
    """
    Log the student's name, timestamp, and emotion to the CSV file.
    Avoid duplicate entries for the same person in one session.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Load existing data if exists
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Name", "Time", "Emotion"])
        
        # Check if already logged in this session
        if not ((df["Name"] == name) & (df["Time"].str.contains(now.strftime("%Y-%m-%d")))).any():
            new_entry = pd.DataFrame([[name, timestamp, emotion]], columns=["Name", "Time", "Emotion"])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Logged attendance for {name} at {timestamp}")
        else:
            print(f"[INFO] {name} already logged today.")
    except Exception as e:
        print(f"[ERROR] Failed to log attendance: {e}")
