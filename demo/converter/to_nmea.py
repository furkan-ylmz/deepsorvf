import pandas as pd
from pyais import encode_dict

# Dosyayı doğru ayraçla oku
df = pd.read_csv("ais_all_sorted_space.txt", sep=r" ", engine="python")

# Gereksiz kolonları at (Unnamed: 0 gibi)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

with open("ais_all_sorted.nmea", "w") as f:
    for _, row in df.iterrows():
        try:
            # timestamp'ten saniye kısmını al
            utc_second = (int(row['timestamp']) // 1000) % 60

            # encode_dict'e dictionary ver
            msg = {
                'type': 1,  # Position Report Class A
                'mmsi': int(row['mmsi']),
                'status': 0,
                'turn': 0,
                'speed': float(row['speed']),
                'accuracy': 1,
                'lon': float(row['lon']),
                'lat': float(row['lat']),
                'course': float(row['course']),
                'heading': int(row['heading']),
                'second': int(utc_second),
                'maneuver': 0,
                'raim': False,
                'radio': 0
            }

            nmea_list = encode_dict(msg)  # Bu bir liste döner (fragmentler)
            for nmea in nmea_list:
                f.write(nmea + "\n")

        except Exception as e:
            print(f"Satır atlandı: {e}")
