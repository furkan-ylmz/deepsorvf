import numpy as np
import pandas as pd
import time

class AutoTimestampSynchronizer:
    """
    Automatic Timestamp Synchronization Engine between Camera Video Frames and AIS Telemetry.
    Uses Normalized Cross-Correlation (NCC) on speed and motion profiles to calculate the
    optimal time offset (tau*) in milliseconds.
    """
    def __init__(self, initial_offset_ms=5 * 3600 * 1000, verify_interval_sec=60):
        self.current_offset_ms = initial_offset_ms
        self.verify_interval_sec = verify_interval_sec
        self.last_sync_timestamp = 0
        self.is_synced = False
        
        self.vis_history = []  # List of dicts: {'timestamp_ms': t, 'speed': s, 'x': x, 'y': y}
        self.ais_history = []  # List of dicts: {'timestamp_ms': t, 'speed': s, 'lat': lat, 'lon': lon}
        self.max_history_len = 300  # ~10-30 seconds of history

    def add_visual_sample(self, timestamp_ms, speed_magnitude):
        """Record visual motion sample."""
        if speed_magnitude is not None and not np.isnan(speed_magnitude):
            self.vis_history.append({
                'timestamp_ms': timestamp_ms,
                'speed': float(speed_magnitude)
            })
            if len(self.vis_history) > self.max_history_len:
                self.vis_history.pop(0)

    def add_ais_sample(self, timestamp_ms, speed_knots):
        """Record AIS speed sample."""
        if speed_knots is not None and not np.isnan(speed_knots):
            self.ais_history.append({
                'timestamp_ms': timestamp_ms,
                'speed': float(speed_knots)
            })
            if len(self.ais_history) > self.max_history_len:
                self.ais_history.pop(0)

    def should_sync(self, current_timestamp_ms):
        """Check if synchronization or periodic verification is needed."""
        if not self.is_synced and len(self.vis_history) >= 60 and len(self.ais_history) >= 20:
            return True
        if self.is_synced and (current_timestamp_ms - self.last_sync_timestamp) >= (self.verify_interval_sec * 1000):
            return True
        return False

    def compute_offset(self, search_window_sec=10, step_ms=200):
        """
        Compute optimal time offset using Normalized Cross-Correlation (NCC).
        
        Returns:
            int: Optimal time offset in milliseconds.
        """
        if len(self.vis_history) < 30 or len(self.ais_history) < 10:
            return self.current_offset_ms

        vis_df = pd.DataFrame(self.vis_history)
        ais_df = pd.DataFrame(self.ais_history)

        # Standardize speeds
        vis_speeds = vis_df['speed'].values
        ais_speeds = ais_df['speed'].values

        if np.std(vis_speeds) < 1e-4 or np.std(ais_speeds) < 1e-4:
            return self.current_offset_ms

        vis_norm = (vis_speeds - np.mean(vis_speeds)) / (np.std(vis_speeds) + 1e-6)
        ais_norm = (ais_speeds - np.mean(ais_speeds)) / (np.std(ais_speeds) + 1e-6)

        # Test lags in search window
        lags = np.arange(-search_window_sec * 1000, search_window_sec * 1000 + 1, step_ms)
        correlations = []

        min_len = min(len(vis_norm), len(ais_norm))
        v_sub = vis_norm[-min_len:]
        a_sub = ais_norm[-min_len:]

        corr = np.correlate(v_sub, a_sub, mode='full')
        best_idx = np.argmax(corr)
        lag_offset = (best_idx - len(a_sub) + 1) * step_ms

        # If correlation peak is meaningful, adjust current offset
        if np.max(corr) > 0.3 * len(v_sub):
            new_offset = self.current_offset_ms + int(lag_offset)
            self.current_offset_ms = new_offset
            self.is_synced = True
            self.last_sync_timestamp = vis_df['timestamp_ms'].iloc[-1]
            print(f"[Auto-Sync] Optimal timestamp offset calibrated: {new_offset} ms (Corr Peak: {np.max(corr):.2f})")

        return self.current_offset_ms
