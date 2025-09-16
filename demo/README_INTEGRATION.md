# Vessel Tracking System - Kurulum ve Kullanım Kılavuzu

DeepSORVF sisteminden çıkarılan AIS, VIS ve FUS modüllerinin yeni bir projede nasıl kullanılacağını anlatan kapsamlı kılavuz.

## 📁 Dosya Yapısı

Aşağıdaki dosyaları hedef projenize kopyalayın:

```
vessel_tracking/
├── ais.py                 # AIS işleme modülü
├── vis.py                 # Görsel takip modülü  
├── fusion.py              # Sensör füzyon modülü
├── vessel_tracker.py      # Birleşik arayüz
├── requirements.txt       # Gerekli paketler
└── example_usage.py       # Kullanım örnekleri
```

## 🔧 Kurulum

### 1. Gerekli Paketlerin Kurulumu

```bash
pip install pandas numpy opencv-python torch torchvision geopy pyproj scipy fastdtw Pillow
```

### 2. Deep Learning Modelleri (Opsiyonel)

VIS modülü için YOLOX ve DeepSORT modelleri:

```bash
# YOLOX kurulumu
pip install yolox

# DeepSORT için ek bağımlılıklar
pip install tensorboard tensorboardX
```

### 3. Kamera Parametreleri

`camera_para.txt` dosyasından kamera parametrelerini alın:
```
[lon, lat, heading, focal_length, camera_height, ...]
```

## 💡 Temel Kullanım

### Basit Örnek

```python
from vessel_tracker import VesselTracker
import cv2

# Konfigürasyon
config = {
    'ais_config': {
        'data_path': '/path/to/ais/data',
        'files': ['ais_file1.csv', 'ais_file2.csv']
    },
    'vis_config': {
        'anti_occlusion': True,
        'detection_model': 'yolox'
    },
    'fusion_config': {
        'max_distance': 200,
        'time_interval': 33
    },
    'camera_config': {
        'image_shape': [1920, 1080],
        'parameters': [114.327, 30.600, 45.0]  # [lon, lat, heading]
    }
}

# Tracker başlatma
tracker = VesselTracker(**config)

# Video işleme
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Frame işleme
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    results = tracker.process_frame(frame, timestamp)
    
    # Sonuçları kullanma
    ais_vessels = results['ais']['current']
    vis_tracks = results['vis']['current'] 
    fused_data = results['fusion']['matched']
    
    print(f"AIS: {len(ais_vessels)}, VIS: {len(vis_tracks)}, Fused: {len(fused_data)}")

cap.release()
```

### Sadece Belirli Modülleri Kullanma

#### Sadece AIS İşleme

```python
from ais import AISPRO

# AIS processor başlatma
ais = AISPRO(
    ais_path='/path/to/ais',
    ais_file='file1.csv',
    im_shape=[1920, 1080],
    t=33
)

# Kamera parametreleri ile işleme
camera_params = [114.327, 30.600, 45.0, 1500, 50, 0.02, 0.65]
timestamp = 1654336512000
time_name = "2022_06_04_12_05_12.csv"

ais_visible, ais_current = ais.process(camera_params, timestamp, time_name)
print(f"Görünen gemiler: {len(ais_visible)}")
```

#### Sadece Görsel Takip

```python
from vis import VISPRO
import cv2

# VIS processor başlatma
vis = VISPRO(
    bin_inf=pd.DataFrame(),
    anti_occlusion=1
)

# Frame işleme
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    ais_data = pd.DataFrame()  # Boş AIS verisi
    bin_inf = pd.DataFrame()   # Boş bağlama verisi
    
    vis_tra, vis_cur = vis.feedCap(frame, timestamp, ais_data, bin_inf)
    print(f"Takip edilen objeler: {len(vis_cur)}")

cap.release()
```

#### Sadece Füzyon

```python
from fusion_demo import Fusion
import pandas as pd

# Fusion processor başlatma
fusion = Fusion(
    max_distance=200,
    image_shape=[1920, 1080],
    time_interval=33
)

# Örnek veri
ais_visible = pd.DataFrame({
    'mmsi': [12345, 67890],
    'x_pixel': [500, 800],
    'y_pixel': [400, 600],
    'timestamp': [1000, 1000]
})

vis_trajectories = pd.DataFrame({
    'ID': [1, 2],
    'x_pixel': [510, 790],
    'y_pixel': [405, 595],
    'timestamp': [1000, 1000]
})

# Füzyon işlemi
fused_data, bin_inf = fusion.fusion(
    ais_visible, pd.DataFrame(), vis_trajectories, pd.DataFrame(), 1000
)
print(f"Eşleşen gemiler: {len(fused_data)}")
```

## 🔗 Modül Arayüzleri

### AISProcessor

```python
class AISProcessor:
    def __init__(self, ais_data_path, ais_files, image_shape, time_interval):
        pass
    
    def process(self, camera_parameters, timestamp, time_name=None):
        """
        Returns:
            ais_visible: Kamera görüş alanındaki gemiler
            ais_current: Mevcut timestamp'teki tüm gemiler
        """
        pass
```

### VISProcessor

```python
class VISProcessor:
    def __init__(self, anti_occlusion, time_interval, detection_model, tracking_config):
        pass
    
    def feedCap(self, frame, timestamp, ais_data, bin_inf):
        """
        Returns:
            vis_trajectories: Tüm takip geçmişi
            vis_current: Mevcut frame'deki tespit/takipler
        """
        pass
```

### Fusion

```python
class Fusion:
    def __init__(self, max_distance, image_shape, time_interval):
        pass
    
    def fusion(self, ais_visible, ais_current, vis_tra, vis_cur, timestamp):
        """
        Returns:
            fused_data: Eşleştirilmiş AIS-VIS verileri
            bin_inf: Bağlama bilgileri (ID-MMSI eşleştirmesi)
        """
        pass
```

## 📊 Veri Formatları

### AIS Verisi

```python
ais_data = pd.DataFrame({
    'mmsi': [int],          # Gemi ID
    'lat': [float],         # Enlem
    'lon': [float],         # Boylam
    'timestamp': [int],     # Zaman damgası (ms)
    'x_pixel': [float],     # Piksel koordinatı X
    'y_pixel': [float],     # Piksel koordinatı Y
    'status': [str]         # Gemi durumu
})
```

### VIS Verisi

```python
vis_data = pd.DataFrame({
    'ID': [int],            # Takip ID
    'x_pixel': [float],     # Piksel koordinatı X
    'y_pixel': [float],     # Piksel koordinatı Y
    'width': [float],       # Bounding box genişlik
    'height': [float],      # Bounding box yükseklik
    'timestamp': [int],     # Zaman damgası (ms)
    'confidence': [float]   # Tespit güveni
})
```

### Füzyon Verisi

```python
fused_data = pd.DataFrame({
    'mmsi': [int],          # AIS gemi ID
    'ID': [int],            # VIS takip ID
    'x_pixel': [float],     # Ortak piksel koordinatı X
    'y_pixel': [float],     # Ortak piksel koordinatı Y
    'timestamp': [int],     # Zaman damgası
    'distance': [float],    # Eşleştirme mesafesi
    'confidence': [float]   # Eşleştirme güveni
})
```

## ⚙️ Konfigürasyon Seçenekleri

### AIS Konfigürasyonu

```python
ais_config = {
    'data_path': '/path/to/ais',     # AIS veri klasörü
    'files': ['file1.csv'],          # İşlenecek dosyalar (boş = otomatik)
    'coordinate_system': 'WGS84',    # Koordinat sistemi
    'filter_distance': 1000,         # Metre cinsinden filtreleme mesafesi
    'interpolation': True            # Zaman interpolasyonu aktif/pasif
}
```

### VIS Konfigürasyonu

```python
vis_config = {
    'anti_occlusion': True,          # Anti-oklüzyon aktif/pasif
    'occlusion_rate': 0.3,           # Oklüzyon eşiği
    'detection_model': 'yolox',      # Tespit modeli (yolox/yolo/custom)
    'tracking_config': {             # DeepSORT ayarları
        'max_dist': 0.2,
        'min_confidence': 0.3,
        'max_iou_distance': 0.7,
        'max_age': 30,
        'n_init': 3
    }
}
```

### Füzyon Konfigürasyonu

```python
fusion_config = {
    'max_distance': 200,             # Maksimum eşleştirme mesafesi (piksel)
    'time_interval': 33,             # Frame aralığı (ms)
    'assignment_method': 'hungarian', # Eşleştirme algoritması
    'dtw_window': 10,                # DTW pencere boyutu
    'trajectory_length': 100         # Maksimum trajektori uzunluğu
}
```

## 🚀 Performans Optimizasyonu

### 1. Modül Seçimi

- Sadece ihtiyacınız olan modülleri kullanın
- AIS yoksa VIS-only mode kullanın
- Real-time gerekmiyorsa batch processing yapın

### 2. Konfigürasyon Ayarları

```python
# Hızlı işleme için
config = {
    'vis_config': {
        'anti_occlusion': False,      # Oklüzyon kontrolünü kapat
        'detection_model': 'simple'   # Basit tespit modeli
    },
    'fusion_config': {
        'max_distance': 100,          # Mesafe eşiğini düşür
        'dtw_window': 5               # DTW penceresini küçült
    }
}

# Yüksek doğruluk için  
config = {
    'vis_config': {
        'anti_occlusion': True,       # Oklüzyon kontrolü aktif
        'detection_model': 'yolox'    # Gelişmiş tespit modeli
    },
    'fusion_config': {
        'max_distance': 300,          # Geniş eşleştirme alanı
        'dtw_window': 20              # Büyük DTW penceresi
    }
}
```

### 3. Bellek Yönetimi

```python
# Periyodik reset
if frame_count % 1000 == 0:
    tracker.reset_tracking()

# Batch processing
for i in range(0, len(frames), batch_size):
    batch = frames[i:i+batch_size]
    # Process batch
```

## 🔍 Hata Ayıklama

### Yaygın Sorunlar

1. **Import Hataları**
   ```bash
   pip install --upgrade pandas numpy opencv-python
   ```

2. **AIS Dosya Formatı**
   ```python
   # CSV dosyaları şu kolonları içermeli:
   # mmsi, lat, lon, timestamp, ...
   ```

3. **Kamera Parametreleri**
   ```python
   # Doğru format: [lon, lat, heading, focal_length, height, ...]
   camera_params = [114.327, 30.600, 45.0, 1500, 50]
   ```

4. **CUDA/GPU Sorunları**
   ```python
   # CPU-only mode için
   vis_config = {
       'tracking_config': {'device': 'cpu'}
   }
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detaylı hata mesajları için
tracker = VesselTracker(debug=True, **config)
```

## 📈 Performans Metrikleri

```python
# Anlık performans
metrics = tracker.get_performance_metrics()
print(f"FPS: {metrics['fps_capability']:.1f}")
print(f"Processing time: {metrics['processing_time_ms']:.2f}ms")

# Gemiler
status = tracker.get_vessel_status()  
print(f"Total vessels: {status['total_count']}")
print(f"Fused matches: {len(status['fused_vessels'])}")
```

## 💾 Sonuç Kaydetme

```python
# CSV format
tracker.save_results('results.csv', format='csv')

# JSON format  
tracker.save_results('results.json', format='json')

# Manuel kayıt
results = tracker.last_results
fusion_data = results['fusion']['matched']
fusion_data.to_csv('my_results.csv', index=False)
```

## 🌐 İleri Seviye Kullanım

### Custom Detection Model

```python
class CustomDetector:
    def detect(self, frame):
        # Custom detection logic
        return detections

vis_config = {
    'detection_model': 'custom',
    'custom_detector': CustomDetector()
}
```

### Multi-Camera Setup

```python
cameras = [
    {'id': 'cam1', 'params': [114.1, 30.5, 45], 'ais_path': '/ais1'},
    {'id': 'cam2', 'params': [114.2, 30.6, 90], 'ais_path': '/ais2'}
]

trackers = {}
for cam in cameras:
    config['camera_config']['parameters'] = cam['params']
    config['ais_config']['data_path'] = cam['ais_path']
    trackers[cam['id']] = VesselTracker(**config)
```

### Real-time Processing

```python
import threading
import queue

def process_video_stream(tracker, frame_queue, result_queue):
    while True:
        try:
            frame, timestamp = frame_queue.get(timeout=1)
            results = tracker.process_frame(frame, timestamp)
            result_queue.put(results)
        except queue.Empty:
            continue

# Threading setup
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue()

thread = threading.Thread(target=process_video_stream, 
                         args=(tracker, frame_queue, result_queue))
thread.start()
```

Bu kılavuz ile DeepSORVF modüllerini yeni projenizde başarıyla kullanabilirsiniz!