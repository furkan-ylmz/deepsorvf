import os
import sys
import time
import json
import cv2
import asyncio
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.VIS_utils import VISPRO
from utils.AIS_utils import AISPRO
from utils.FUS_utils import FUSPRO
from utils.draw import DRAW
from utils.file_read import read_all, ais_initial, update_time
from utils.stream_simulator import StreamSimulator
from performance_monitor import PerformanceMonitor

app = FastAPI(title="DeepSORVF Web C2 Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# System State Global Container
class SystemEngine:
    def __init__(self):
        self.mode = "file"  # "file" or "live"
        self.model_name = "yolov8x.pt"
        self.is_running = True
        self.data_path = "./clip-01/"
        self.result_path = "./result/"
        
        # Read parameters
        self.video_path, self.ais_path, _, _, self.initial_time, self.camera_para = read_all(self.data_path, self.result_path)
        self.ais_file, self.timestamp0, _ = ais_initial(self.ais_path, self.initial_time)
        self.time_state = self.initial_time.copy()
        
        self.im_shape = [1920, 1080]
        self.t = 33  # ~30 FPS
        self.max_dis = 200
        
        # Initialize Core Modules
        self.AIS = AISPRO(self.ais_path, self.ais_file, self.im_shape, self.t)
        self.VIS = VISPRO(anti=1, val=0, t=self.t, model_name=self.model_name)
        self.FUS = FUSPRO(self.max_dis, self.im_shape, self.t)
        self.DRA = DRAW(self.im_shape, self.t)
        
        self.simulator = StreamSimulator(self.data_path, self.initial_time)
        self.bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        
        # Performance Monitor
        self.monitor = PerformanceMonitor(log_file="web_c2_performance.csv")
        self.monitor.start_monitoring()
        
        self.current_frame = None
        self.current_overlay = None
        self.latest_telemetry = {}
        self.lock = threading.Lock()
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def set_model(self, new_model_name):
        with self.lock:
            self.model_name = new_model_name
            self.VIS.set_model(new_model_name)

    def set_mode(self, new_mode):
        with self.lock:
            self.mode = new_mode
            if new_mode == "file":
                self.simulator.reset()

    def _process_loop(self):
        fps_counter = 0
        fps_start = time.time()
        current_fps = 30.0

        while self.is_running:
            start_t = time.time()
            
            ret, frame, timestamp, time_name = self.simulator.get_next_frame()
            if not ret or frame is None:
                self.simulator.reset()
                continue
                
            self.im_shape = [frame.shape[1], frame.shape[0]]
            
            # 1. AIS Processing
            AIS_vis, AIS_cur = self.AIS.process(self.camera_para, timestamp, time_name)
            
            # 2. VIS Processing (YOLOv8/v11 + ByteTrack)
            Vis_tra, Vis_cur = self.VIS.feedCap(frame, timestamp, AIS_vis, self.bin_inf)
            
            # 3. Fusion Processing (FastDTW + Hungarian)
            Fus_tra, self.bin_inf = self.FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
            
            # 4. Draw Overlay
            overlay = self.DRA.draw_traj(frame.copy(), AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp, self.camera_para)
            
            # FPS Calculation
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
                
            # Telemetry Construction
            ais_list = []
            for _, r in AIS_cur.iterrows():
                try:
                    cx, cy = AISPRO.visual_transform(r['lon'], r['lat'], self.camera_para, self.im_shape)
                    ais_list.append({
                        "mmsi": int(r['mmsi']), "lon": float(r['lon']), "lat": float(r['lat']),
                        "speed": float(r['speed']), "course": float(r['course']), "x": cx, "y": cy
                    })
                except Exception:
                    pass

            fusion_matches = []
            for _, r in Fus_tra.iterrows():
                fusion_matches.append({
                    "vis_id": int(r['ID']), "mmsi": int(r['mmsi']),
                    "speed": float(r['speed']), "lat": float(r['lat']), "lon": float(r['lon'])
                })

            vis_tracks = []
            for _, r in Vis_cur.iterrows():
                vis_tracks.append({
                    "id": int(r['ID']), "x1": int(r['x1']), "y1": int(r['y1']),
                    "x2": int(r['x2']), "y2": int(r['y2'])
                })

            # Detect Unidentified Targets (Dark Ships)
            unmatched_vis = [t for t in vis_tracks if not any(f['vis_id'] == t['id'] for f in fusion_matches)]

            with self.lock:
                self.current_frame = frame
                self.current_overlay = overlay
                self.latest_telemetry = {
                    "timestamp": timestamp,
                    "time_name": time_name,
                    "fps": round(current_fps, 1),
                    "mode": self.mode,
                    "model": self.model_name,
                    "ais_count": len(ais_list),
                    "vis_count": len(vis_tracks),
                    "fusion_count": len(fusion_matches),
                    "unmatched_count": len(unmatched_vis),
                    "ais_vessels": ais_list,
                    "vis_tracks": vis_tracks,
                    "fusion_matches": fusion_matches,
                    "unmatched_vis": unmatched_vis,
                    "camera_para": self.camera_para[:2]  # [lon_cam, lat_cam]
                }
                
            elapsed = time.time() - start_t
            sleep_t = max(0.01, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_t)

engine = SystemEngine()

@app.get("/")
def read_root():
    return HTMLResponse(content=open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8").read())

@app.get("/video_feed")
def video_feed():
    """MJPEG Video Streamer for frontend HTML5 image player."""
    def generate():
        while True:
            with engine.lock:
                if engine.current_overlay is None:
                    time.sleep(0.03)
                    continue
                # Resize for responsive web playback
                resized = cv2.resize(engine.current_overlay, (960, 540))
                _, jpeg = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with engine.lock:
                data = engine.latest_telemetry
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)  # 10 Hz Telemetry update
    except WebSocketDisconnect:
        pass

@app.post("/api/model")
async def set_model(request: Request):
    data = await request.json()
    model_name = data.get("model", "yolov8x.pt")
    engine.set_model(model_name)
    return {"status": "ok", "model": model_name}

@app.post("/api/mode")
async def set_mode(request: Request):
    data = await request.json()
    mode = data.get("mode", "file")
    engine.set_mode(mode)
    return {"status": "ok", "mode": mode}

@app.get("/api/status")
def get_status():
    with engine.lock:
        return engine.latest_telemetry

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
