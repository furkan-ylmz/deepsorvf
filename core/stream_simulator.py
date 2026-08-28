import cv2
import time
import os
import glob
import pandas as pd
from core.data_loader import time2stamp, update_time

class StreamSimulator:
    """
    Offline Stream Simulator: Replays video files and AIS CSV logs in real-time or fast mode.
    Simulates IP Camera (RTSP) and AIS Socket feeds for testing.
    """
    def __init__(self, data_path="./data/", initial_time=None):
        self.data_path = data_path
        video_files = glob.glob(os.path.join(data_path, "*.mp4")) + glob.glob(os.path.join(data_path, "*.avi"))
        if len(video_files) == 0:
            raise FileNotFoundError(f"No video file found in {data_path}")
            
        self.video_path = video_files[0]
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_delay_ms = int(1000 / self.fps)
        
        self.initial_time = initial_time or [2022, 6, 4, 12, 5, 12, 0]
        self.current_time = self.initial_time.copy()
        self.frame_count = 0

    def get_next_frame(self):
        """
        Fetch next simulated frame and updated timestamp.
        
        Returns:
            tuple: (ret, frame, timestamp_ms, time_name)
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, 0, ""
            
        self.current_time, timestamp_ms, time_name = update_time(self.current_time, self.frame_delay_ms)
        self.frame_count += 1
        return True, frame, timestamp_ms, time_name

    def reset(self):
        """Reset video capture to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_time = self.initial_time.copy()
        self.frame_count = 0

    def close(self):
        self.cap.release()
