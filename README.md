# 🚢 DeepSORVF - Modernized Vessel Tracking & AIS Fusion

**DeepSORVF** (*Deep-learning, Spatial-temporal Optimal Matching, Relative Vector Fusion*), gemi kameralarından alınan video akışları ile **AIS (Automatic Identification System)** telsiz verilerini gerçek zamanlı ve çevrimdışı ortamda birleştirerek gemi Görsel Takip ID'leri ile AIS MMSI numaralarını eşleştiren gelişmiş bir sensör füzyonu sistemidir.

---

## 🌟 Modernize Edilmiş Özellikler (`dev` Branch)

- **Ultralytics YOLOv8 / YOLO11 & PyTorch CUDA:** YOLOX yerine CUDA donanım ivmelendirmeli YOLOv8/v11 modelleri (`yolov8n` .. `yolov8x`).
- **ByteTrack & BoT-SORT Çoklu Nesne Takibi:** Düşük güvenilirlikli tespitleri koruyan 2 aşamalı IoU ve Kalman filtresi ile kararlı ID takibi.
- **Yerel Web Komuta Kontrol Arayüzü (Web C2 Dashboard):**
  - **FastAPI & WebSocket:** Canlı video akışı (MJPEG) ve 10 Hz telemetry yayını.
  - **Leaflet.js & OpenSeaMap:** Canlı uydu ve denizcilik haritası katmanları.
  - **Karanlık Hedef / Tanımsız Gemi İkazı (Dark Ship Alert):** AIS ile eşleşmeyen tanımsız gemiler için kırmızı yanıp sönen ikaz paneli.
- **Çift Modlu Akış Mimarisi (Dual Stream Architecture):**
  - **File Mode:** Arşiv MP4 ve CSV verilerini replayer ile çalıştırma ve Ground Truth (`clip-01/gt`) ile MOTA/IDF1 benchmark testi.
  - **Live Web Mode:** YouTube 7/24 Deniz Kameraları + `aisstream.io` ücretsiz WebSocket API'si ile sıfır maliyetli canlı akış.

---

## 📂 Temiz & Modüler Dizin Yapısı

```
deepsorvf/
├── main.py                  # CLI çalıştırma ve iş akışı giriş noktası
├── performance_monitor.py   # CPU, RAM, GPU ve Watt güç tüketim izleyici
├── requirements.txt         # Proje bağımlılıkları
├── utils/                   # Çekirdek İşlem ve Füzyon Paketi
│   ├── yolo_detector.py     # Ultralytics YOLOv8/v11 PyTorch CUDA dedektörü
│   ├── bytetrack_tracker.py # ByteTrack / BoT-SORT çoklu nesne takipçisi
│   ├── AIS_utils.py         # AIS jeodezik projeksiyon ve zaman senkronizasyonu
│   ├── VIS_utils.py         # Görsel işleme ve Anti-Occlusion mantığı
│   ├── FUS_utils.py         # Multi-Feature FastDTW ve Hungarian eşleştirici
│   ├── draw.py              # Video overlay ve bilgi paneli çizen modül
│   ├── stream_simulator.py  # Offline arşiv replayer akış simülatörü
│   ├── live_stream.py        # YouTube Live + aisstream.io canlı akış bağlayıcı
│   ├── file_read.py         # Dosya okuma ve konfigürasyon yardımcısı
│   └── gen_result.py        # Sonuç CSV metrik yazıcı
├── web_app/                 # Yerel Web Komuta Kontrol Paneli (Web C2)
│   ├── server.py            # FastAPI & WebSocket backend sunucusu
│   └── static/              # HTML5, CSS ve Leaflet.js frontend dosyaları
├── tests/
│   └── evaluate.py          # Ground Truth MOTA, IDF1 benchmark test betiği
└── clip-01/                 # Örnek veri seti (Video, AIS CSV'leri, GT)
```

---

## 🚀 Hızlı Başlangıç

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Yerel Web Komuta Kontrol Panelini (Web C2) Başlatın
```bash
python web_app/server.py
```
Tarayıcınızda **`http://localhost:8000`** adresini açarak Komuta Kontrol Paneline ulaşabilirsiniz.

### 3. CLI İle Çevrimdışı Çalıştırma
```bash
python main.py --data_path ./clip-01/ --monitor
```

### 4. Ground Truth Metrik Testi
```bash
python tests/evaluate.py
```