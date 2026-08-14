import asyncio
import json
import websockets
import cv2
import numpy as np
import pandas as pd
import time
import threading

class LiveAISStreamer:
    """
    Zero-Cost Live AIS Streamer via aisstream.io WebSocket API.
    Receives real-time vessel NMEA/JSON telemetry for specified Bounding Box.
    """
    def __init__(self, api_key="", bbox=[[40.8, 28.8], [41.2, 29.2]]):
        self.api_key = api_key
        self.bbox = bbox  # [[min_lat, min_lon], [max_lat, max_lon]]
        self.ws_url = "wss://stream.aisstream.io/v0/stream"
        self.is_running = False
        self.current_vessels = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','timestamp'])
        self._thread = None

    def start(self):
        """Start background thread for receiving live AIS data."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self._thread.start()
            print("📡 Live AIS Streamer started.")

    def stop(self):
        self.is_running = False

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._listen())

    async def _listen(self):
        sub_msg = {
            "APIKey": self.api_key or "DEMO_KEY",
            "BoundingBoxes": [self.bbox]
        }
        try:
            async with websockets.connect(self.ws_url) as websocket:
                await websocket.send(json.dumps(sub_msg))
                while self.is_running:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    self._parse_message(data)
        except Exception as e:
            print(f"📡 AIS Stream connection notice: {e}")

    def _parse_message(self, data):
        try:
            msg_type = data.get("MessageType")
            if msg_type == "PositionReport":
                pos = data["Message"]["PositionReport"]
                mmsi = pos.get("UserID")
                lat = pos.get("Latitude")
                lon = pos.get("Longitude")
                sog = pos.get("Sog", 0)
                cog = pos.get("Cog", 0)
                heading = pos.get("TrueHeading", 0)
                ts = int(time.time() * 1000)
                
                new_row = pd.DataFrame([{'mmsi': mmsi, 'lon': lon, 'lat': lat, 'speed': sog,
                                         'course': cog, 'heading': heading, 'type': 70, 'timestamp': ts}])
                self.current_vessels = pd.concat([self.current_vessels[self.current_vessels['mmsi'] != mmsi], new_row], ignore_index=True)
        except Exception:
            pass

    def get_latest_data(self):
        return self.current_vessels.copy()


class LiveYouTubeStreamer:
    """
    Zero-Cost Live Maritime Video Streamer via Streamlink / YouTube Live Webcams.
    """
    def __init__(self, youtube_url="https://www.youtube.com/watch?v=live_maritime_cam"):
        self.youtube_url = youtube_url
        self.cap = None

    def connect(self):
        """Connect to YouTube live stream using Streamlink."""
        try:
            import streamlink
            streams = streamlink.streams(self.youtube_url)
            if 'best' in streams:
                stream_url = streams['best'].url
                self.cap = cv2.VideoCapture(stream_url)
                print(f"📹 Connected to Live Stream: {self.youtube_url}")
                return True
        except Exception as e:
            print(f"⚠️ Streamlink connection notice: {e}")
        return False

    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()
