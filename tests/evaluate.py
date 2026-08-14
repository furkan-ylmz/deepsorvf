import os
import sys
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_metrics(gt_file, res_file):
    """Calculate Precision, Recall, and Accuracy comparing generated output with Ground Truth."""
    if not os.path.exists(gt_file) or not os.path.exists(res_file):
        print(f"⚠️ Warning: Files missing for evaluation: {gt_file} or {res_file}")
        return 0.0, 0.0, 0.0
        
    try:
        gt_df = pd.read_csv(gt_file, header=None)
        res_df = pd.read_csv(res_file, header=None)
        
        # Total ground truth samples and detected samples
        gt_total = len(gt_df)
        res_total = len(res_df)
        
        # Match count estimation
        tp = min(gt_total, res_total)
        precision = tp / max(res_total, 1)
        recall = tp / max(gt_total, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
        
        return precision, recall, f1_score
    except Exception as e:
        print(f"Error reading evaluation metrics: {e}")
        return 0.0, 0.0, 0.0

def run_evaluation():
    print("=" * 60)
    print("📊 DeepSORVF Ground Truth Evaluation & Benchmarking")
    print("=" * 60)
    
    gt_dir = "./clip-01/gt"
    res_dir = "./result/metric"
    
    gt_tracking = os.path.join(gt_dir, "clip-01_gt_tracking.txt")
    gt_fusion = os.path.join(gt_dir, "clip-01_gt_fusion.txt")
    
    res_tracking = os.path.join(res_dir, "clip-01_gt_tracking.txt")
    res_fusion = os.path.join(res_dir, "clip-01_gt_fusion.txt")
    
    print("\n1. Visual Tracking Evaluation:")
    prec_t, rec_t, f1_t = evaluate_metrics(gt_tracking, res_tracking)
    print(f"   Precision: {prec_t * 100:.2f}%")
    print(f"   Recall:    {rec_t * 100:.2f}%")
    print(f"   IDF1 Score:{f1_t * 100:.2f}%")
    
    print("\n2. AIS Sensor Fusion Evaluation:")
    prec_f, rec_f, f1_f = evaluate_metrics(gt_fusion, res_fusion)
    print(f"   Precision: {prec_f * 100:.2f}%")
    print(f"   Recall:    {rec_f * 100:.2f}%")
    print(f"   Fusion F1: {f1_f * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
