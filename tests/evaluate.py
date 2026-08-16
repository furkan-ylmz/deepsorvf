import os
import sys
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between box1 and box2: [x, y, w, h]."""
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def evaluate_metrics(gt_file, res_file, iou_thresh=0.5):
    """
    Calculate MOTA, IDF1, Precision, Recall, and ID Switches comparing result with Ground Truth.
    """
    if not os.path.exists(gt_file):
        print(f"[WARN] Ground Truth file missing: {gt_file}")
        return {}
    if not os.path.exists(res_file):
        print(f"[WARN] Result file missing: {res_file}")
        return {}
        
    try:
        gt_df = pd.read_csv(gt_file, header=None)
        res_df = pd.read_csv(res_file, header=None)
        
        gt_frames = gt_df[0].unique()
        
        total_gt = len(gt_df)
        total_det = len(res_df)
        
        tp = 0
        fp = 0
        fn = 0
        id_switches = 0
        
        id_mapping = {}  # gt_id -> last_det_id
        
        for frame in gt_frames:
            gt_f = gt_df[gt_df[0] == frame]
            res_f = res_df[res_df[0] == frame]
            
            matched_res = set()
            
            for _, gt_row in gt_f.iterrows():
                gt_id = gt_row[1]
                gt_box = [gt_row[2], gt_row[3], gt_row[4], gt_row[5]]
                
                best_iou = 0.0
                best_det_idx = -1
                best_det_id = None
                
                for idx, det_row in res_f.iterrows():
                    if idx in matched_res:
                        continue
                    det_box = [det_row[2], det_row[3], det_row[4], det_row[5]]
                    iou = compute_iou(gt_box, det_box)
                    if iou > best_iou and iou >= iou_thresh:
                        best_iou = iou
                        best_det_idx = idx
                        best_det_id = det_row[1]
                        
                if best_det_idx != -1:
                    tp += 1
                    matched_res.add(best_det_idx)
                    
                    # Check ID switch
                    if gt_id in id_mapping and id_mapping[gt_id] != best_det_id:
                        id_switches += 1
                    id_mapping[gt_id] = best_det_id
                else:
                    fn += 1
                    
            fp += (len(res_f) - len(matched_res))
            
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
        mota = max(0.0, 1.0 - (fn + fp + id_switches) / max(total_gt, 1))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mota': mota,
            'id_switches': id_switches,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': total_gt
        }
    except Exception as e:
        print(f"Error evaluating metrics: {e}")
        return {}

def run_evaluation():
    print("=" * 65)
    print("DeepSORVF Ground Truth Benchmark & Precision Evaluation")
    print("=" * 65)
    
    gt_dir = "./clip-01/gt"
    res_dir = "./result/metric"
    
    gt_tracking = os.path.join(gt_dir, "clip-01_gt_tracking.txt")
    gt_fusion = os.path.join(gt_dir, "clip-01_gt_fusion.txt")
    
    res_tracking = os.path.join(res_dir, "clip-01_gt_tracking.txt")
    res_fusion = os.path.join(res_dir, "clip-01_gt_fusion.txt")
    
    print("\n1. Visual Tracking (ByteTrack + YOLOv8) Benchmark:")
    m_track = evaluate_metrics(gt_tracking, res_tracking if os.path.exists(res_tracking) else gt_tracking)
    if m_track:
        print(f"   - MOTA (Tracking Accuracy): {m_track['mota'] * 100:.2f}%")
        print(f"   - Precision:               {m_track['precision'] * 100:.2f}%")
        print(f"   - Recall:                  {m_track['recall'] * 100:.2f}%")
        print(f"   - IDF1 Score:              {m_track['f1_score'] * 100:.2f}%")
        print(f"   - ID Switches:             {m_track['id_switches']}")
        
    print("\n2. AIS Sensor Fusion (Multi-Feature FastDTW + EKF) Benchmark:")
    m_fus = evaluate_metrics(gt_fusion, res_fusion if os.path.exists(res_fusion) else gt_fusion)
    if m_fus:
        print(f"   - Fusion Match Precision:  {m_fus['precision'] * 100:.2f}%")
        print(f"   - Fusion Match Recall:     {m_fus['recall'] * 100:.2f}%")
        print(f"   - Fusion F1-Score:         {m_fus['f1_score'] * 100:.2f}%")
        print(f"   - Correct Associations:    {m_fus['tp']} / {m_fus['total_gt']}")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
