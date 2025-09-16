# 🚢 DeepSORVF Vessel Tracking - Kullanım Rehberi

## 📁 Dosya Yapısı (Temiz)

```
vessel_tracking/
├── ais.py              # AIS işleme modülü
├── vis.py              # Görsel takip modülü  
├── fusion.py           # Sensör füzyon modülü
├── vessel_tracker.py   # Birleşik arayüz (opsiyonel)
├── example_usage.py    # Örnekler (opsiyonel)
└── requirements.txt    # Gerekli paketler
```

## 🔧 Kurulum

### 1. Dosyaları Kopyala
```bash
# Bu dosyaları projenize kopyalayın:
ais.py
vis.py
fusion.py
vessel_tracker.py  # (opsiyonel - birleşik kullanım için)
```

### 2. Bağımlılıkları Kur
```bash
pip install pandas numpy opencv-python geopy pyproj scipy fastdtw

# VIS modülü için (opsiyonel):
pip install torch torchvision yolox
```

## 💡 Nasıl Kullanılır?

### Seçenek 1: Ayrı Modüller Halinde

#### 🗺️ **AIS Modülü (ais.py)**

```python
from ais import AISPRO

# Başlatma
ais = AISPRO(
    ais_path="./data/ais/",           # AIS dosyalarının bulunduğu klasör
    ais_file="ais_data.csv",          # AIS dosya adı
    im_shape=[1920, 1080],            # Kamera çözünürlüğü
    t=[lon, lat, heading, ...]        # Kamera parametreleri
)

# Kullanım
time_name = "2022_06_04_12_05_30"    # Zaman damgası
ais_vis, ais_cur = ais.GetAisData(time_name)

# Çıktı:
# ais_vis: Kamera görüş alanındaki gemiler (x,y koordinatları ile)
# ais_cur: Mevcut tüm AIS verileri
```

#### 📹 **VIS Modülü (vis.py)**

```python
from vis import VISPRO
import cv2

# Başlatma
vis = VISPRO(
    bin_inf=[x, y, w, h],            # İlgi alanı (opsiyonel)
    anti_occlusion=True              # Anti-occlusion aktif
)

# Kullanım
frame = cv2.imread("frame.jpg")       # Video karesi
timestamp = "2022_06_04_12_05_30"     # Zaman damgası
ais_data = [...] # AIS verisi         # AIS modülünden gelen veri
bin_inf = [0, 0, 1920, 1080]         # İlgi alanı

vis_tra, vis_cur = vis.feedCap(frame, timestamp, ais_data, bin_inf)

# Çıktı:
# vis_tra: Takip edilen gemiler (trajectory)
# vis_cur: Mevcut tespit edilen gemiler
```

#### 🔗 **Fusion Modülü (fusion.py)**

```python
from fusion import Fusion

# Başlatma
fusion = Fusion(
    max_dis=200,                     # Maksimum eşleştirme mesafesi (pixel)
    time_inter=33                    # Zaman aralığı (ms)
)

# Kullanım
ais_data = [...]                     # AIS verisi
vis_data = [...]                     # VIS verisi

fused_data = fusion.data_fusion(ais_data, vis_data)

# Çıktı:
# fused_data: Eşleştirilmiş AIS+VIS verileri
```

### Seçenek 2: Birleşik Kullanım (vessel_tracker.py)

```python
from vessel_tracker import VesselTracker

# Ayarları tanımla
ais_config = {
    'data_path': './data/ais/',
    'files': ['ais_2022_06_04.csv']
}

vis_config = {
    'anti_occlusion': True,
    'detection_model': 'yolox'
}

fusion_config = {
    'max_distance': 200,
    'time_interval': 33
}

camera_config = {
    'image_shape': [1920, 1080],
    'parameters': [lon, lat, heading, focal_length, ...]
}

# Tracker'ı başlat
tracker = VesselTracker(
    ais_config=ais_config,
    vis_config=vis_config,
    fusion_config=fusion_config,
    camera_config=camera_config
)

# Tek frame işle
import cv2
frame = cv2.imread("frame.jpg")
timestamp = "2022_06_04_12_05_30"

results = tracker.process_frame(frame, timestamp)

# Çıktı:
print(f"AIS gemileri: {len(results['ais_vessels'])}")
print(f"VIS takipleri: {len(results['vis_tracks'])}")
print(f"Fusion sonucu: {len(results['fused_data'])}")
```

## 📊 Çıktı Formatları

### AIS Çıktısı
```python
ais_vis = pd.DataFrame({
    'mmsi': [123456789, 987654321],           # Gemi ID
    'lon': [28.9765, 28.9823],                # Boylam
    'lat': [41.0082, 41.0156],                # Enlem
    'speed': [12.5, 8.3],                     # Hız (knot)
    'course': [45.2, 180.0],                  # Rota (derece)
    'heading': [50.1, 185.2],                 # Yön (derece)
    'type': [70, 36],                         # Gemi tipi
    'x': [856, 1024],                         # Ekran X koordinatı
    'y': [432, 567],                          # Ekran Y koordinatı
    'timestamp': ['2022_06_04_12_05_30', ...] # Zaman damgası
})
```

### VIS Çıktısı
```python
vis_tra = {
    'track_1': {
        'id': 1,
        'bbox': [x, y, w, h],                 # Sınırlayıcı kutu
        'center': [cx, cy],                   # Merkez koordinat
        'confidence': 0.95,                   # Güven skoru
        'class': 'ship',                      # Nesne sınıfı
        'trajectory': [[x1,y1], [x2,y2], ...]  # Hareket yolu
    },
    'track_2': {...}
}
```

### Fusion Çıktısı
```python
fused_data = [
    {
        'ais_id': 123456789,                  # AIS MMSI
        'vis_id': 1,                          # VIS track ID
        'matched': True,                      # Eşleşme durumu
        'distance': 45.2,                     # Eşleşme mesafesi
        'confidence': 0.87,                   # Eşleşme güveni
        'position': [x, y],                   # Final pozisyon
        'ais_data': {...},                    # AIS detayları
        'vis_data': {...}                     # VIS detayları
    }
]
```

## 🔧 Neler Gerekli?

### Minimum Gereksinimler
1. **Python 3.7+**
2. **Temel paketler**: pandas, numpy, opencv
3. **AIS için**: geopy, pyproj
4. **Fusion için**: scipy, fastdtw

### VIS için Ek Gereksinimler
1. **Deep Learning**: torch, torchvision
2. **YOLOX modeli**: Gemi tespiti için
3. **DeepSORT**: Çoklu nesne takibi için

### Veri Gereksinimleri

#### AIS Verisi (CSV format)
```csv
mmsi,lon,lat,speed,course,heading,type,timestamp
123456789,28.9765,41.0082,12.5,45.2,50.1,70,2022-06-04 12:05:30
```

#### Kamera Parametreleri
```python
camera_params = [
    longitude,          # Kamera boylam
    latitude,           # Kamera enlem  
    heading,            # Kamera yönü (derece)
    focal_length,       # Odak uzaklığı
    camera_height,      # Kamera yüksekliği
    # ... diğer parametreler
]
```

## 🚀 Hızlı Başlangıç

### 1. Sadece AIS İşleme
```python
from ais import AISPRO

ais = AISPRO("./ais/", "data.csv", [1920,1080], camera_params)
ais_vis, ais_cur = ais.GetAisData("2022_06_04_12_05_30")
print(f"Görünen gemiler: {len(ais_vis)}")
```

### 2. Video İşleme
```python
import cv2
from vessel_tracker import VesselTracker

tracker = VesselTracker(ais_config, vis_config, fusion_config, camera_config)

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    timestamp = "2022_06_04_12_05_30"  # Gerçek timestamp kullan
    results = tracker.process_frame(frame, timestamp)
    
    # Sonuçları işle
    for vessel in results['fused_data']:
        if vessel['matched']:
            print(f"Eşleşen gemi: AIS {vessel['ais_id']} <-> VIS {vessel['vis_id']}")
```

## ⚠️ Önemli Notlar

1. **VIS modülü** YOLOX modelini gerektirir
2. **Kamera parametreleri** doğru ayarlanmalıdır
3. **AIS verisi** zaman damgası ile senkronize olmalıdır
4. **Fusion** hem AIS hem VIS verisini gerektirir

## 🔍 Hata Giderme

```python
# Test et
python example_usage.py

# Eksik modülleri kontrol et
try:
    from ais import AISPRO
    print("✅ AIS OK")
except:
    print("❌ AIS hatası")

try:
    from vis import VISPRO
    print("✅ VIS OK") 
except:
    print("❌ VIS hatası - YOLOX gerekli")
```