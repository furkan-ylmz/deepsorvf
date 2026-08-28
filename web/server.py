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

from core.vis_processor import VISPRO
from core.ais_processor import AISPRO
from core.fusion_processor import FUSPRO
from core.visualizer import DRAW
from core.data_loader import read_all, ais_initial, update_time
from core.stream_simulator import StreamSimulator
from core.time_sync import AutoTimestampSynchronizer
from core.camera_profiles import MARITIME_PROFILES, get_profile, list_profiles, calculate_camera_parameters
from core.live_stream import LiveAISStreamer, LiveYouTubeStreamer

app = FastAPI(title="DeepSORVF Web C2 Dashboard", version="2.5.0")

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

class SystemEngine:
    """
    Core Execution Engine managing Dual Stream Modes (File Replayer & Zero-Cost Live Web Stream),
    YOLOv8 Detection, ByteTrack Tracking, Multi-Feature DTW, EKF Fusion, and Live WebSocket Telemetry.
    """
    def __init__(self, data_path="./data/", result_path="./result/"):
        self.data_path = data_path
        self.result_path = result_path
        self.model_name = "yolov8x.pt"
        self.mode = "file"  # "file" or "live"
        self.is_running = True
        
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
        self.time_sync = AutoTimestampSynchronizer()
        
        self.simulator = StreamSimulator(self.data_path, self.initial_time)
        self.bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        
        # Live Web Stream Components
        self.active_profile_id = "istanbul_bosphorus"
        self.api_key = ""
        self.live_ais = LiveAISStreamer(bbox=get_profile("istanbul_bosphorus")["ais_bbox"])
        self.live_video = LiveYouTubeStreamer()
        self.calibration_state = {
            "heading": 45.0, "pitch": -5.0, "roll": 0.0, "height": 35.0, "fov": 45.0
        }
        
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
                self.live_video.stop()
                self.live_ais.stop()
                self.simulator.reset()
                # Restore dataset camera para
                _, _, _, _, _, self.camera_para = read_all(self.data_path, self.result_path)

    def connect_live(self, profile_id="istanbul_bosphorus", api_key="", custom_url="", custom_gps=None):
        """Connect to real-world live stream and matching AIS stream."""
        with self.lock:
            self.mode = "live"
            self.active_profile_id = profile_id
            if api_key:
                self.api_key = api_key
                
            prof = get_profile(profile_id)
            stream_url = custom_url.strip() if (custom_url and custom_url.strip()) else prof.get("youtube_url", "")
            bbox = prof["ais_bbox"]
            
            self.camera_para = prof["camera_para"].copy()
            self.calibration_state = {
                "heading": prof["heading_deg"],
                "pitch": prof["pitch_deg"],
                "roll": prof["roll_deg"],
                "height": prof["camera_height_m"],
                "fov": prof["fov_deg"]
            }
            
            self.live_ais.update_config(api_key=self.api_key, bbox=bbox)
            self.live_video.set_url(stream_url)
            self.live_video.start()
            self.live_ais.start()
            print(f"[INFO] Live Mode Connected: {prof['name']} (URL: {stream_url})")

    def calibrate_camera(self, heading, pitch, height, fov, roll=0.0):
        """Dynamically update camera calibration parameters on the running feed."""
        with self.lock:
            self.calibration_state = {
                "heading": float(heading),
                "pitch": float(pitch),
                "roll": float(roll),
                "height": float(height),
                "fov": float(fov)
            }
            lon_cam, lat_cam = self.camera_para[0], self.camera_para[1]
            self.camera_para = calculate_camera_parameters(
                lon_cam, lat_cam, float(height), float(heading), float(pitch), float(roll), float(fov),
                self.im_shape[0], self.im_shape[1]
            )

    def _create_placeholder_frame(self, profile_name="İstanbul Boğazı", ais_count=0):
        img = np.zeros((self.im_shape[1], self.im_shape[0], 3), dtype=np.uint8)
        # Background gradient: Dark sky and ocean
        img[0:int(self.im_shape[1]*0.55), :] = (25, 20, 15)  # Dark night sky
        img[int(self.im_shape[1]*0.55):, :] = (40, 28, 18)   # Sea water
        
        # Horizon Line
        cv2.line(img, (0, int(self.im_shape[1]*0.55)), (self.im_shape[0], int(self.im_shape[1]*0.55)), (0, 180, 220), 1)
        
        # Tactical HUD Crosshair
        cx, cy = self.im_shape[0] // 2, self.im_shape[1] // 2
        cv2.line(img, (cx - 40, cy), (cx + 40, cy), (0, 210, 255), 1)
        cv2.line(img, (cx, cy - 40), (cx, cy + 40), (0, 210, 255), 1)
        cv2.circle(img, (cx, cy), 20, (0, 210, 255), 1)
        
        # Grid range rings
        cv2.circle(img, (cx, cy), int(self.im_shape[1] * 0.35), (45, 65, 85), 1)
        cv2.circle(img, (cx, cy), int(self.im_shape[1] * 0.45), (35, 50, 65), 1)
        
        # Top HUD Text
        cv2.putText(img, f"[TACTICAL C2 VIEWPORT: {profile_name.upper()}]", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 118), 2, cv2.LINE_AA)
        cv2.putText(img, f"CANLI AIS HEDEFLERI: {ais_count} GEMI | STATUS: LIVE TELEMETRY", (40, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 1, cv2.LINE_AA)
        
        # Bottom HUD Text
        cal = self.calibration_state
        cv2.putText(img, f"KALIBRASYON: YON {cal.get('heading', 45)} deg | EGIM {cal.get('pitch', -5)} deg | YUKSEKLIK {cal.get('height', 35)}m | FOV {cal.get('fov', 45)} deg",
                    (40, self.im_shape[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return img

    def _process_loop(self):
        fps_counter = 0
        fps_start = time.time()
        current_fps = 30.0

        while self.is_running:
            start_t = time.time()
            
            if self.mode == "live":
                # LIVE WEB STREAM MODE
                raw_ais_df = self.live_ais.get_latest_data()
                AIS_cur = raw_ais_df.copy()
                
                ret, frame = self.live_video.read_frame()
                if not ret or frame is None:
                    prof_name = get_profile(self.active_profile_id).get("name", "İstanbul Boğazı")
                    frame = self._create_placeholder_frame(profile_name=prof_name, ais_count=len(AIS_cur))
                
                timestamp = int(time.time() * 1000)
                time_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(timestamp / 1000.0))
                self.im_shape = [frame.shape[1], frame.shape[0]]
                AIS_vis = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'speed', 'course', 'heading', 'type', 'x', 'y', 'timestamp'])
                AIS_cur = raw_ais_df.copy()
                
                # Project AIS coordinates to screen pixels
                for _, row in AIS_cur.iterrows():
                    try:
                        cx, cy = AISPRO.visual_transform(row['lon'], row['lat'], self.camera_para, self.im_shape)
                        if 0 <= cx < self.im_shape[0] and 0 <= cy < self.im_shape[1]:
                            v_row = row.to_dict()
                            v_row['x'] = cx
                            v_row['y'] = cy
                            AIS_vis = pd.concat([AIS_vis, pd.DataFrame([v_row])], ignore_index=True)
                    except Exception:
                        pass
            else:
                # FILE REPLAYER MODE
                ret, frame, timestamp, time_name = self.simulator.get_next_frame()
                if not ret or frame is None:
                    self.simulator.reset()
                    continue
                    
                self.im_shape = [frame.shape[1], frame.shape[0]]
                AIS_vis, AIS_cur = self.AIS.process(self.camera_para, timestamp, time_name)
            
            # 2. Visual Detection and Tracking (YOLOv8/v11 + ByteTrack)
            Vis_tra, Vis_cur = self.VIS.feedCap(frame, timestamp, AIS_vis, self.bin_inf)

            # 3. Auto-Sync Time Offset Calculation (Normalized Cross-Correlation)
            for _, r in Vis_cur.iterrows():
                if 'speed' in r:
                    try:
                        s_val = str(r['speed'])
                        if ',' in s_val:
                            vx, vy = [float(x) for x in s_val.strip("[]()").split(",")]
                            self.time_sync.add_visual_sample(timestamp, (vx**2 + vy**2)**0.5)
                    except Exception:
                        pass

            for _, r in AIS_cur.iterrows():
                if 'speed' in r:
                    try:
                        self.time_sync.add_ais_sample(timestamp, float(r['speed']))
                    except Exception:
                        pass

            if self.time_sync.should_sync(timestamp):
                opt_offset = self.time_sync.compute_offset()
                self.AIS.set_time_offset(opt_offset)
            
            # 4. Fusion Processing (Multi-Feature FastDTW + Adaptive Hungarian + EKF)
            Fus_tra, self.bin_inf = self.FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
            
            # 5. Draw Visual Trajectory & Vessel Overlay
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
                except Exception:
                    cx, cy = -1, -1
                    
                ais_list.append({
                    "mmsi": int(r['mmsi']),
                    "name": str(r.get('name', f"MMSI_{r['mmsi']}")),
                    "lon": float(r['lon']), "lat": float(r['lat']),
                    "speed": float(r['speed']), "course": float(r['course']), "x": cx, "y": cy
                })

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
            ekf_states = self.FUS.get_ekf_states()

            with self.lock:
                self.current_frame = frame
                self.current_overlay = overlay
                self.latest_telemetry = {
                    "timestamp": timestamp,
                    "time_name": time_name,
                    "fps": round(current_fps, 1),
                    "mode": self.mode,
                    "model": self.model_name,
                    "profile_id": self.active_profile_id,
                    "ais_connected": self.live_ais.is_connected if self.mode == "live" else True,
                    "video_connected": self.live_video.is_connected if self.mode == "live" else True,
                    "ais_count": len(ais_list),
                    "vis_count": len(vis_tracks),
                    "fusion_count": len(fusion_matches),
                    "unmatched_count": len(unmatched_vis),
                    "ais_vessels": ais_list,
                    "vis_tracks": vis_tracks,
                    "fusion_matches": fusion_matches,
                    "unmatched_vis": unmatched_vis,
                    "ekf_tracks": ekf_states,
                    "calibration": self.calibration_state,
                    "time_offset_ms": self.time_sync.current_offset_ms,
                    "camera_para": self.camera_para[:2]  # [lon_cam, lat_cam]
                }
                
            elapsed = time.time() - start_t
            sleep_t = max(0.01, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_t)

engine = SystemEngine()

@app.get("/")
def read_root():
    return HTMLResponse(
        content=open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8").read(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/video_feed")
def video_feed():
    """MJPEG Video Streamer for frontend HTML5 image player."""
    def generate():
        while True:
            with engine.lock:
                if engine.current_overlay is None:
                    time.sleep(0.03)
                    continue
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

@app.get("/api/live/profiles")
def get_live_profiles():
    """List available maritime camera profiles."""
    return list_profiles()

@app.post("/api/live/connect")
async def connect_live(request: Request):
    """Connect to live camera and AIS stream."""
    data = await request.json()
    profile_id = data.get("profile_id", "istanbul_bosphorus")
    api_key = data.get("api_key", "")
    custom_url = data.get("custom_url", "")
    engine.connect_live(profile_id=profile_id, api_key=api_key, custom_url=custom_url)
    return {"status": "ok", "profile_id": profile_id}

@app.post("/api/live/calibrate")
async def calibrate_camera(request: Request):
    """Calibrate camera angles dynamically."""
    data = await request.json()
    heading = data.get("heading", 45.0)
    pitch = data.get("pitch", -5.0)
    height = data.get("height", 35.0)
    fov = data.get("fov", 45.0)
    engine.calibrate_camera(heading=heading, pitch=pitch, height=height, fov=fov)
    return {"status": "ok", "calibration": engine.calibration_state}

@app.get("/api/status")
def get_status():
    with engine.lock:
        return engine.latest_telemetry

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
