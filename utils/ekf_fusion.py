import numpy as np
import pandas as pd
import math
import time

class VesselEKFFilter:
    """
    Extended Kalman Filter for individual fused vessel trajectory tracking.
    State Vector: X = [x, y, vx, vy, w, h]^T
    Fuses high-frequency camera measurements (30 Hz) with asynchronous AIS telemetry (0.5 - 1 Hz).
    Supports 5-second Coasting Mode when AIS signal drops.
    """
    def __init__(self, vis_id, mmsi, init_x, init_y, init_w=50, init_h=30, init_vx=0.0, init_vy=0.0):
        self.vis_id = vis_id
        self.mmsi = mmsi
        
        # State: [center_x, center_y, vx, vy, width, height]
        self.x = np.array([float(init_x), float(init_y), float(init_vx), float(init_vy), float(init_w), float(init_h)], dtype=np.float64)
        
        # Initial Covariance Matrix P
        self.P = np.diag([10.0, 10.0, 5.0, 5.0, 15.0, 15.0]).astype(np.float64)
        
        # Process Noise Q parameters
        self.q_pos = 1.0
        self.q_vel = 0.5
        self.q_dim = 0.1
        
        # Measurement Noise Covariances
        self.R_vis = np.diag([4.0, 4.0, 8.0, 8.0]).astype(np.float64)  # [x, y, w, h]
        self.R_ais = np.diag([15.0, 15.0, 2.0, 2.0]).astype(np.float64)  # [x, y, vx, vy]
        
        self.last_ais_update_time = time.time()
        self.last_vis_update_time = time.time()
        self.max_coast_seconds = 5.0
        self.is_active = True
        self.seniority = 1  # Continuous match score

    def predict(self, dt=0.033):
        """Predict next state using constant velocity motion model."""
        dt = max(0.001, min(dt, 1.0))
        
        # State Transition Matrix F
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Process Noise Covariance Q
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0, 0] = self.q_pos * (dt**3) / 3.0
        Q[1, 1] = self.q_pos * (dt**3) / 3.0
        Q[2, 2] = self.q_vel * dt
        Q[3, 3] = self.q_vel * dt
        Q[4, 4] = self.q_dim * dt
        Q[5, 5] = self.q_dim * dt
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update_vis(self, bbox_xywh):
        """Measurement update from Camera Vision detection [x, y, w, h]."""
        z_vis = np.array([float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])], dtype=np.float64)
        
        H_vis = np.zeros((4, 6), dtype=np.float64)
        H_vis[0, 0] = 1.0
        H_vis[1, 1] = 1.0
        H_vis[2, 4] = 1.0
        H_vis[3, 5] = 1.0
        
        # Kalman Gain
        S = H_vis @ self.P @ H_vis.T + self.R_vis
        K = self.P @ H_vis.T @ np.linalg.inv(S)
        
        # Innovation
        y = z_vis - H_vis @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H_vis) @ self.P
        
        self.last_vis_update_time = time.time()
        self.seniority += 1

    def update_ais(self, ais_x, ais_y, ais_vx=None, ais_vy=None):
        """Measurement update from AIS telemetry."""
        if ais_vx is None or ais_vy is None:
            # Position-only AIS update
            z_ais = np.array([float(ais_x), float(ais_y)], dtype=np.float64)
            H_ais = np.zeros((2, 6), dtype=np.float64)
            H_ais[0, 0] = 1.0
            H_ais[1, 1] = 1.0
            R = self.R_ais[:2, :2]
        else:
            z_ais = np.array([float(ais_x), float(ais_y), float(ais_vx), float(ais_vy)], dtype=np.float64)
            H_ais = np.zeros((4, 6), dtype=np.float64)
            H_ais[0, 0] = 1.0
            H_ais[1, 1] = 1.0
            H_ais[2, 2] = 1.0
            H_ais[3, 3] = 1.0
            R = self.R_ais
            
        S = H_ais @ self.P @ H_ais.T + R
        K = self.P @ H_ais.T @ np.linalg.inv(S)
        y = z_ais - H_ais @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H_ais) @ self.P
        
        self.last_ais_update_time = time.time()

    def get_coast_duration(self):
        """Return elapsed seconds since last direct AIS message."""
        return time.time() - self.last_ais_update_time

    def is_coasting(self):
        """Check if vessel is in Coast Mode (AIS temporarily missing but within 5s)."""
        coast_sec = self.get_coast_duration()
        return 0.5 < coast_sec <= self.max_coast_seconds

    def is_coast_expired(self):
        """True if AIS signal has been missing for more than 5 seconds."""
        return self.get_coast_duration() > self.max_coast_seconds

    def get_state(self):
        """Return smoothed state dictionary."""
        cx, cy, vx, vy, w, h = self.x
        speed_pix = math.sqrt(vx**2 + vy**2)
        heading_rad = math.atan2(vy, vx) if speed_pix > 0.01 else 0.0
        heading_deg = (math.degrees(heading_rad) + 360) % 360
        
        return {
            'vis_id': self.vis_id,
            'mmsi': self.mmsi,
            'x': float(cx),
            'y': float(cy),
            'x1': float(cx - w/2),
            'y1': float(cy - h/2),
            'x2': float(cx + w/2),
            'y2': float(cy + h/2),
            'w': float(w),
            'h': float(h),
            'vx': float(vx),
            'vy': float(vy),
            'speed_pix': float(speed_pix),
            'heading_deg': float(heading_deg),
            'is_coasting': self.is_coasting(),
            'seniority': self.seniority
        }


class EKFFusionManager:
    """
    Manager for all active Vessel EKF instances.
    Coordinates prediction, sensor updates, coasting lifecycle, and smoothed trajectory output.
    """
    def __init__(self):
        self.filters = {}  # Key: f"{vis_id}/{mmsi}" -> VesselEKFFilter
        self.last_predict_time = time.time()

    def predict_all(self):
        """Advance all filters to current timestamp."""
        now = time.time()
        dt = now - self.last_predict_time
        self.last_predict_time = now
        
        for key, ekf in list(self.filters.items()):
            ekf.predict(dt)
            # Remove stale filters inactive for > 10 seconds
            if now - ekf.last_vis_update_time > 10.0:
                del self.filters[key]

    def update_fusion(self, matched_pairs_df, ais_current_df, vis_current_df):
        """
        Update EKFs with latest matched pairs, AIS data, and visual bounding boxes.
        """
        self.predict_all()
        
        for _, match in matched_pairs_df.iterrows():
            vis_id = int(match['ID'])
            mmsi = int(match['mmsi'])
            key = f"{vis_id}/{mmsi}"
            
            # Find visual detection bbox
            vis_rows = vis_current_df[vis_current_df['ID'] == vis_id]
            if len(vis_rows) > 0:
                v = vis_rows.iloc[-1]
                cx = float(v.get('x', (v['x1'] + v['x2']) / 2))
                cy = float(v.get('y', (v['y1'] + v['y2']) / 2))
                w = float(v.get('w', v['x2'] - v['x1']))
                h = float(v.get('h', v['y2'] - v['y1']))
            else:
                cx, cy, w, h = float(match['x1'] + match['w']/2), float(match['y1'] + match['h']/2), float(match['w']), float(match['h'])

            # Create new EKF if not existing
            if key not in self.filters:
                self.filters[key] = VesselEKFFilter(vis_id, mmsi, cx, cy, w, h)
                
            ekf = self.filters[key]
            ekf.update_vis([cx, cy, w, h])
            
            # Update AIS if available
            ais_rows = ais_current_df[ais_current_df['mmsi'] == mmsi]
            if len(ais_rows) > 0 and 'x' in ais_rows.columns:
                a = ais_rows.iloc[-1]
                if not np.isnan(a['x']) and not np.isnan(a['y']):
                    ekf.update_ais(a['x'], a['y'])

    def get_smoothed_tracks(self):
        """Return list of smoothed states for all active vessels."""
        states = []
        for key, ekf in self.filters.items():
            if not ekf.is_coast_expired():
                states.append(ekf.get_state())
        return states
