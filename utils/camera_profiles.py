import math

MARITIME_PROFILES = {
    "istanbul_bosphorus": {
        "id": "istanbul_bosphorus",
        "name": "Istanbul Bosphorus (Sarayburnu / South Entrance)",
        "location": "Istanbul, Turkey",
        "youtube_url": "",
        "camera_gps": [28.9850, 41.0185],  # [lon, lat]
        "camera_height_m": 35.0,
        "heading_deg": 45.0,
        "pitch_deg": -5.0,
        "roll_deg": 0.0,
        "fov_deg": 45.0,
        "ais_bbox": [[[40.85, 28.80], [41.30, 29.30]]],  # [[lat_min, lon_min], [lat_max, lon_max]]
        "camera_para": [28.9850, 41.0185, 45.0, -5.0, 0.0, 35.0, 45.0, 2400.0, 2400.0, 960.0, 540.0]
    },
    "port_rotterdam": {
        "id": "port_rotterdam",
        "name": "Port of Rotterdam (Nieuwe Waterweg / Hook of Holland)",
        "location": "Rotterdam, Netherlands",
        "youtube_url": "",
        "camera_gps": [4.1200, 51.9850],  # [lon, lat]
        "camera_height_m": 25.0,
        "heading_deg": 280.0,
        "pitch_deg": -4.0,
        "roll_deg": 0.0,
        "fov_deg": 40.0,
        "ais_bbox": [[[51.93, 4.05], [52.05, 4.25]]],
        "camera_para": [4.1200, 51.9850, 280.0, -4.0, 0.0, 25.0, 40.0, 2500.0, 2500.0, 960.0, 540.0]
    },
    "port_miami": {
        "id": "port_miami",
        "name": "PortMiami (Government Cut Cruise Channel)",
        "location": "Miami, Florida, USA",
        "youtube_url": "",
        "camera_gps": [-80.1350, 25.7650],  # [lon, lat]
        "camera_height_m": 30.0,
        "heading_deg": 95.0,
        "pitch_deg": -6.0,
        "roll_deg": 0.0,
        "fov_deg": 50.0,
        "ais_bbox": [[[25.74, -80.16], [25.79, -80.10]]],
        "camera_para": [-80.1350, 25.7650, 95.0, -6.0, 0.0, 30.0, 50.0, 2300.0, 2300.0, 960.0, 540.0]
    },
    "istanbul_kizkulesi": {
        "id": "istanbul_kizkulesi",
        "name": "İstanbul Boğazı (Kız Kulesi / Maiden's Tower Live)",
        "location": "Istanbul, Turkey",
        "youtube_url": "https://www.youtube.com/watch?v=ggSqYAd4Xq8",
        "camera_gps": [28.9850, 41.0185],  # [lon, lat]
        "camera_height_m": 35.0,
        "heading_deg": 118.0,
        "pitch_deg": -3.5,
        "roll_deg": 0.0,
        "fov_deg": 45.0,
        "ais_bbox": [[[40.85, 28.80], [41.30, 29.30]]],
        "camera_para": [28.9850, 41.0185, 118.0, -3.5, 0.0, 35.0, 45.0, 2400.0, 2400.0, 960.0, 540.0]
    },
    "custom": {
        "id": "custom",
        "name": "Özel YouTube / RTSP Canlı Akış Linki",
        "location": "User Defined",
        "youtube_url": "",
        "camera_gps": [28.9850, 41.0185],
        "camera_height_m": 35.0,
        "heading_deg": 118.0,
        "pitch_deg": -3.5,
        "roll_deg": 0.0,
        "fov_deg": 45.0,
        "ais_bbox": [[[40.85, 28.80], [41.30, 29.30]]],
        "camera_para": [28.9850, 41.0185, 118.0, -3.5, 0.0, 35.0, 45.0, 2400.0, 2400.0, 960.0, 540.0]
    }
}

def get_profile(profile_id):
    """Retrieve maritime camera profile by ID."""
    return MARITIME_PROFILES.get(profile_id, MARITIME_PROFILES["istanbul_bosphorus"])

def list_profiles():
    """List all available maritime camera profiles."""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "location": p["location"],
            "youtube_url": p["youtube_url"],
            "camera_gps": p["camera_gps"],
            "heading_deg": p["heading_deg"],
            "pitch_deg": p["pitch_deg"],
            "camera_height_m": p["camera_height_m"]
        }
        for p in MARITIME_PROFILES.values()
    ]

def calculate_camera_parameters(lon, lat, height_m, heading_deg, pitch_deg, roll_deg=0.0, fov_deg=45.0, im_w=1920, im_h=1080):
    """
    Compute full 11-element camera calibration parameter vector for pinhole projection.
    [lon, lat, heading, pitch, roll, height, fov, fx, fy, cx, cy]
    """
    fov_rad = math.radians(fov_deg)
    fx = (im_w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = im_w / 2.0
    cy = im_h / 2.0
    
    return [
        float(lon), float(lat), float(heading_deg), float(pitch_deg), float(roll_deg),
        float(height_m), float(fov_deg), float(fx), float(fy), float(cx), float(cy)
    ]
