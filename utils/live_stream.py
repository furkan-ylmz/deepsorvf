import os
import asyncio
import json
import websockets
import cv2
import numpy as np
import pandas as pd
import time
import threading

# Force single-threaded FFmpeg decoding in OpenCV to prevent pthread_frame.c race conditions
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"

class LiveAISStreamer:
    """
    Zero-Cost Live AIS Streamer via aisstream.io WebSocket API.
    Receives real-time vessel NMEA/JSON telemetry for a specified Bounding Box.
    Includes robust auto-reconnect, ping/pong keepalive, and single-thread safety.
    """
    def __init__(self, api_key="", bbox=[[[40.85, 28.80], [41.30, 29.30]]]):
        self.api_key = api_key
        self.bbox = bbox  # [[[min_lat, min_lon], [max_lat, max_lon]]]
        self.ws_url = "wss://stream.aisstream.io/v0/stream"
        self.is_running = False
        self.is_connected = False
        self.current_vessels = pd.DataFrame(columns=['mmsi', 'name', 'lon', 'lat', 'speed', 'course', 'heading', 'type', 'timestamp'])
        self._thread = None
        self.lock = threading.Lock()

    def update_config(self, api_key=None, bbox=None):
        """Update API Key or Bounding Box dynamically without restarting thread."""
        with self.lock:
            if api_key is not None:
                self.api_key = api_key.strip()
            if bbox is not None:
                self.bbox = bbox

    def start(self):
        """Start background thread for receiving live AIS data."""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
                self._thread.start()
                print("[INFO] Live AIS Streamer thread started.")

    def stop(self):
        with self.lock:
            self.is_running = False
            self.is_connected = False

    def _run_async_loop(self):
        while self.is_running:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._listen())
            except Exception as e:
                err_str = str(e)
                print(f"[INFO] AIS Stream status: {err_str}")
                self.is_connected = False
                if "429" in err_str:
                    print("[INFO] aisstream.io single-connection limit (HTTP 429). Waiting 10s...")
                    time.sleep(10)
                else:
                    time.sleep(4)

    async def _listen(self):
        if not self.api_key:
            while self.is_running and not self.api_key:
                await asyncio.sleep(2)
            if not self.is_running:
                return

        sub_msg = {
            "APIKey": self.api_key,
            "BoundingBoxes": self.bbox,
            "FilterMessageTypes": ["PositionReport", "StandardClassBPositionReport", "ShipStaticData"]
        }
        
        print(f"[INFO] Connecting to aisstream.io with BoundingBox: {self.bbox}")
        async with websockets.connect(self.ws_url, ping_interval=15, ping_timeout=15, close_timeout=10) as websocket:
            await websocket.send(json.dumps(sub_msg))
            self.is_connected = True
            print("[INFO] Connected to aisstream.io live WebSocket stream successfully.")
            
            while self.is_running:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(msg)
                    self._parse_message(data)
                except asyncio.TimeoutError:
                    pong_waiter = await websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10.0)

    def _parse_message(self, data):
        try:
            msg_type = data.get("MessageType")
            meta = data.get("MetaData", {})
            mmsi = meta.get("MMSI") or meta.get("mmsi")
            ship_name = (meta.get("ShipName") or meta.get("ship_name") or "").strip()
            
            lat = meta.get("latitude") or meta.get("Latitude")
            lon = meta.get("longitude") or meta.get("Longitude")
            ts = int(time.time() * 1000)
            
            sog = 0.0
            cog = 0.0
            heading = 0.0
            ship_type = 70  # Default Cargo
            
            if msg_type == "PositionReport":
                pos = data.get("Message", {}).get("PositionReport", {})
                sog = float(pos.get("Sog", 0.0))
                cog = float(pos.get("Cog", 0.0))
                heading = float(pos.get("TrueHeading", 0.0))
                if lat is None: lat = pos.get("Latitude") or pos.get("latitude")
                if lon is None: lon = pos.get("Longitude") or pos.get("longitude")
                
            elif msg_type == "StandardClassBPositionReport":
                pos = data.get("Message", {}).get("StandardClassBPositionReport", {})
                sog = float(pos.get("Sog", 0.0))
                cog = float(pos.get("Cog", 0.0))
                heading = float(pos.get("TrueHeading", 0.0))
                if lat is None: lat = pos.get("Latitude") or pos.get("latitude")
                if lon is None: lon = pos.get("Longitude") or pos.get("longitude")
                
            elif msg_type == "ShipStaticData":
                static = data.get("Message", {}).get("ShipStaticData", {})
                ship_type = static.get("Type", 70)
                ship_name = (static.get("Name") or ship_name).strip()

            if mmsi and lat is not None and lon is not None:
                new_row = pd.DataFrame([{
                    'mmsi': int(mmsi),
                    'name': str(ship_name) if ship_name else f"MMSI_{mmsi}",
                    'lon': float(lon),
                    'lat': float(lat),
                    'speed': float(sog),
                    'course': float(cog),
                    'heading': float(heading),
                    'type': int(ship_type),
                    'timestamp': ts
                }])
                print(f"[AIS STREAM] Vessel: {ship_name or mmsi} at ({float(lat):.4f}, {float(lon):.4f}) SOG: {sog} kn")
                
                with self.lock:
                    self.current_vessels = pd.concat([
                        self.current_vessels[self.current_vessels['mmsi'] != int(mmsi)],
                        new_row
                    ], ignore_index=True)
                    
                    # Remove stale vessels not updated for > 120 seconds
                    cutoff_ts = ts - 120000
                    self.current_vessels = self.current_vessels[self.current_vessels['timestamp'] >= cutoff_ts]
        except Exception:
            pass

    def get_latest_data(self):
        with self.lock:
            return self.current_vessels.copy()


class LiveYouTubeStreamer:
    """
    High-Performance Zero-Cost Live Maritime Video Streamer.
    Uses Streamlink to resolve live HLS streams with automatic token refresh and single-thread safety.
    Falls back to sample maritime footage if stream URL is connecting or empty.
    """
    def __init__(self, youtube_url="", fallback_video="./clip-01/2022_06_04_12_05_12_12_07_02_b.mp4"):
        self.youtube_url = youtube_url
        self.fallback_video = fallback_video
        self.cap = None
        self.latest_frame = None
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self._thread = None
        self.current_playing_url = ""

    def set_url(self, new_url):
        """Update stream URL without creating duplicate threads."""
        with self.lock:
            self.youtube_url = new_url.strip()

    def start(self):
        """Idempotent single-thread start."""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()

    def stop(self):
        with self.lock:
            self.is_running = False
            self.is_connected = False

    def _resolve_stream_url(self, url):
        """Resolve YouTube live stream or RTSP/HLS URL."""
        if not url:
            return None
        if url.startswith("rtsp://") or url.endswith(".m3u8") or url.endswith(".mp4"):
            return url
        try:
            import streamlink
            session = streamlink.Streamlink()
            session.set_option("http-headers", "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            streams = session.streams(url)
            if 'best' in streams:
                return streams['best'].url
            elif '720p' in streams:
                return streams['720p'].url
            elif streams:
                return list(streams.values())[0].url
        except Exception as e:
            print(f"[WARN] Streamlink notice: {e}")
        return None

    def _capture_loop(self):
        while self.is_running:
            with self.lock:
                target_url = self.youtube_url
            
            if target_url:
                stream_url = self._resolve_stream_url(target_url)
                if stream_url:
                    try:
                        self.cap = cv2.VideoCapture(stream_url)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        if self.cap.isOpened():
                            self.is_connected = True
                            print(f"[INFO] Live video capture stream connected successfully.")
                            consecutive_failures = 0
                            
                            while self.is_running:
                                with self.lock:
                                    if self.youtube_url != target_url:
                                        break
                                
                                ret, frame = self.cap.read()
                                if ret and frame is not None:
                                    consecutive_failures = 0
                                    with self.lock:
                                        self.latest_frame = frame
                                    time.sleep(0.01)
                                else:
                                    consecutive_failures += 1
                                    if consecutive_failures > 30:
                                        print("[INFO] Live stream HLS chunk expired/stalled. Reconnecting...")
                                        break
                                    time.sleep(0.03)
                    except Exception as e:
                        print(f"[INFO] Live video capture error: {e}")
                    finally:
                        if self.cap:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                        self.is_connected = False
                        time.sleep(2)
                else:
                    time.sleep(2)
            else:
                # If no custom live URL given or while connecting, stream fallback video in loop
                try:
                    self.cap = cv2.VideoCapture(self.fallback_video)
                    self.is_connected = False
                    while self.is_running:
                        with self.lock:
                            if self.youtube_url:
                                break
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        with self.lock:
                            self.latest_frame = frame
                        time.sleep(0.033)
                except Exception:
                    time.sleep(1)
                finally:
                    if self.cap:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None

    def read_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return False, None
