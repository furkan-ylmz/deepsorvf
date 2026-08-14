import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLODetector:
    """
    Ultralytics YOLOv8 / YOLO11 Object Detector Wrapper with PyTorch CUDA Acceleration.
    Supports dynamic model size switching (yolov8n .. yolov8x, yolo11x).
    """
    def __init__(self, model_name="yolov8x.pt", conf_thresh=0.25, iou_thresh=0.45, device=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Auto-detect PyTorch CUDA device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 YOLODetector Initializing | Model: {model_name} | Device: {self.device}")
        self.model_name = model_name
        self.model = YOLO(model_name)
        
        # COCO class index for boat/ship
        self.target_classes = [8]  # 8: 'boat' in COCO dataset

    def change_model(self, model_name):
        """Dynamically switch YOLO model weights at runtime."""
        if model_name != self.model_name:
            print(f"🔄 Switching YOLO model from {self.model_name} to {model_name}...")
            self.model_name = model_name
            self.model = YOLO(model_name)
            print(f"✅ YOLO Model switched to {model_name}")

    def detect(self, image):
        """
        Run YOLO detection on input image (numpy array BGR format from OpenCV).
        
        Returns:
            list: List of detections in format [(x1, y1, x2, y2, 'vessel', conf), ...]
        """
        if image is None:
            return []
            
        results = self.model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            classes=self.target_classes,
            verbose=False
        )
        
        bboxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                bboxes.append((x1, y1, x2, y2, 'vessel', conf))
                
        return bboxes
