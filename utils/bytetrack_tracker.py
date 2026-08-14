import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def iou(box1, box2):
    """Calculate IoU between box1 [x1,y1,x2,y2] and box2 [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class Track:
    def __init__(self, track_id, bbox, score):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.age = 1
        self.time_since_update = 0

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score
        self.age += 1
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1
        self.age += 1

class ByteTracker:
    """
    ByteTrack algorithm implementation for robust vessel multi-object tracking.
    Retains low-confidence detections and uses two-stage IoU association.
    """
    def __init__(self, high_thresh=0.4, low_thresh=0.1, match_thresh=0.3, max_time_lost=30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.next_id = 1

    def update(self, bboxes, bboxes_anti_occ, id_list, timestamp):
        """
        Update tracker with high & low confidence detection boxes.
        
        Returns:
            list: Active tracks [(x1, y1, x2, y2, 'vessel', track_id), ...]
        """
        all_boxes = list(bboxes) + list(bboxes_anti_occ)
        
        # Predict track positions
        for track in self.tracked_tracks:
            track.predict()
            
        # Divide detections into high score and low score
        det_high = [b for b in all_boxes if b[5] >= self.high_thresh]
        det_low = [b for b in all_boxes if self.low_thresh <= b[5] < self.high_thresh]
        
        # --- Stage 1: Association with high-confidence detections ---
        matched_high, unmatch_tracks, unmatch_det_high = self._associate(
            self.tracked_tracks, det_high, self.match_thresh
        )
        
        for t_idx, d_idx in matched_high:
            self.tracked_tracks[t_idx].update(det_high[d_idx][:4], det_high[d_idx][5])
            
        # --- Stage 2: Association unmatched tracks with low-confidence detections ---
        remain_tracks = [self.tracked_tracks[i] for i in unmatch_tracks]
        matched_low, unmatch_tracks_low, _ = self._associate(
            remain_tracks, det_low, self.match_thresh
        )
        
        for t_idx, d_idx in matched_low:
            remain_tracks[t_idx].update(det_low[d_idx][:4], det_low[d_idx][5])
            
        # Create new tracks for unmatched high-confidence detections
        for d_idx in unmatch_det_high:
            b = det_high[d_idx]
            new_track = Track(self.next_id, b[:4], b[5])
            self.next_id += 1
            self.tracked_tracks.append(new_track)
            
        # Remove dead tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if t.time_since_update <= self.max_time_lost]
        
        output_tracks = []
        for track in self.tracked_tracks:
            if track.time_since_update == 0:
                x1, y1, x2, y2 = track.bbox
                output_tracks.append((x1, y1, x2, y2, 'vessel', track.track_id))
                
        return output_tracks

    def _associate(self, tracks, detections, thresh):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
            
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                cost_matrix[i, j] = 1.0 - iou(t.bbox, d[:4])
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - thresh):
                matched.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
                    
        return matched, unmatched_tracks, unmatched_dets
