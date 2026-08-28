# DeepSORVF - Gemi Takip ve AIS Sensör Füzyon Sistemi

DeepSORVF (Deep learning-enabled Asynchronous Trajectory Matching-based Vessel Data Fusion), gemi kameralarından alınan görsel video akışları ile AIS (Automatic Identification System) telsiz verilerini zaman-mekansal optimal eşleştirme algoritmaları kullanarak birleştiren hibrit bir denizcilik takip ve kimliklendirme sistemidir.

Bu proje, kamera görüntüsündeki gemi tespitlerine (YOLO) ve çoklu nesne takibine (ByteTrack) karşılık gelen AIS MMSI numaralarını, hız, rota ve enlem/boylam bilgilerini eşleştirerek görsel ve coğrafi veriyi tek bir komuta kontrol panelinde sunar.

---

## Sistem Mimarisi ve Temel Bileşenler

Sistem mimarisi dört ana işlem modülünden oluşmaktadır:

1. **VISPRO (Görsel Algılama ve Takip Modülü):**
   - Ultralytics YOLOv8 ve YOLO11 mimarileri kullanılarak PyTorch CUDA ivmelendirmesi ile gemi tespiti yapılır.
   - ByteTrack ve BoT-SORT algoritmaları ile nesnelerin kareler arası kimlik (ID) sürekliliği korunur.
   - Gemi kapanmaları (occlusion) durumunda geçmiş hareket vektörlerinden faydalanılarak Anti-Occlusion tahmini üretilir.

2. **AISPRO (AIS İşleme ve Coğrafi Projeksiyon Modülü):**
   - AIS telsiz verilerindeki geçersiz koordinat ve mantıksız hız sıçramaları filtrelenir.
   - PyProj WGS84 jeodezik ileri öngörü formülleri ile eksik zaman damgaları tahmin edilir.
   - Kamera açısı, yüksekliği, odak uzaklığı ve bakış yönü kullanılarak 3D coğrafi koordinatlar (Lat, Lon) 2D piksel koordinatlarına (X, Y) dönüştürülür.

3. **FUSPRO (Sensör Füzyon ve Atama Modülü):**
   - Görsel iz zaman serileri ile AIS projeksiyon izleri arasındaki yön açısı farkı ve mesafe hesaplanır.
   - FastDTW (Fast Dynamic Time Warping) algoritması kullanılarak zaman-hizalamalı benzerlik maliyet matrisi oluşturulur.
   - Macar (Hungarian / Linear Sum Assignment) algoritması ile küresel minimum maliyetli Görsel ID ↔ AIS MMSI eşleştirmesi çözülür.

4. **Web C2 (Yerel Komuta Kontrol Arayüzü):**
   - FastAPI ve WebSocket mimarisi ile 30 FPS canlı video akışı ve 10 Hz telemetri yayını sağlanır.
   - Leaflet.js altyapısında OpenStreetMap, Esri World Imagery ve OpenSeaMap denizcilik harita katmanları sunulur.
   - AIS yayını yapmayan veya eşleşmeyen gemiler için Tanımsız Hedef (Dark Ship) ikaz sistemi çalışır.

---

## Teknolojik Altyapı

- **Programlama Dili:** Python 3.10+
- **Derin Öğrenme ve Takip:** Ultralytics (YOLOv8 / YOLO11), PyTorch (CUDA İvmelendirmeli), ByteTrack
- **Görüntü İşleme:** OpenCV, NumPy, Pillow, Imutils
- **Coğrafi ve Matematiksel Hesaplama:** PyProj, GeoPy, SciPy, FastDTW
- **Web ve Arayüz:** FastAPI, Uvicorn, WebSockets, HTML5, CSS3, JavaScript, Leaflet.js, OpenSeaMap
- **Sistem İzleme:** Psutil, GPUtil, Pynvml

---

## Sistem Mimarisi ve Akış Şemaları

![DeepSORVF Sistem Akış Şeması](docs/flowchart.png)

![DeepSORVF Sıralı İşlem Diyagramı](docs/sequence_diagram.png)

---

## Proje Dizin Yapısı

```
deepsorvf/
├── data/                    # Örnek test veri seti (Video, AIS CSV verileri, GT)
├── docs/                    # Mimari akış ve sıralı işlem şemaları (PNG)
│   ├── flowchart.png
│   └── sequence_diagram.png
├── models/                  # Ultralytics YOLOv8 ve YOLO11 ağırlık dosyaları
│   ├── yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt / yolo11x.pt
├── result/                  # Olay yakalama ve analiz çıktıları (.gitignore)
├── tests/                   # Benchmark doğrulama testleri
│   └── test_benchmark.py    # Ground Truth MOTA, IDF1 benchmark doğrulama betiği
├── core/                    # Çekirdek İşlem ve Füzyon Paketi
│   ├── ais_processor.py     # AIS jeodezik hesaplama ve coğrafi projeksiyon
│   ├── vis_processor.py     # Görsel tespit, takip ve Anti-Occlusion
│   ├── fusion_processor.py  # Multi-Feature FastDTW ve Hungarian eşleştirici
│   ├── ekf_fusion.py        # Genişletilmiş Kalman Filtresi (EKF) ve 5s Coasting
│   ├── time_sync.py         # NCC Otomatik Zaman Senkronizasyonu
│   ├── yolo_detector.py     # Ultralytics YOLOv8/v11 PyTorch CUDA dedektörü
│   ├── byte_tracker.py      # ByteTrack / BoT-SORT çoklu nesne takipçisi
│   ├── visualizer.py        # Video üzerine HUD çerçeve ve bilgi paneli çizimi
│   ├── live_stream.py       # YouTube Live + aisstream.io canlı akış bağlayıcı
│   ├── camera_profiles.py   # Kız Kulesi, Boğaz, Rotterdam hazır kamera profilleri
│   ├── stream_simulator.py  # Offline arşiv replayer akış simülatörü
│   └── data_loader.py       # Veri seti okuma ve konfigürasyon yardımcısı
├── web/                     # Yerel Web Komuta Kontrol Paneli (Web C2)
│   ├── server.py            # FastAPI sunucusu ve WebSocket telemetri motoru
│   └── static/              # HTML, CSS ve JavaScript frontend dosyaları
│       ├── index.html       # Komuta kontrol ana ekranı
│       ├── css/style.css    # Dark-mode C2 arayüz tasarımı
│       └── js/app.js        # Leaflet harita ve WebSocket istemci mantığı
├── requirements.txt         # Proje bağımlılık listesi
├── README.md                # Proje dokümantasyonu
└── .gitignore               # Git dışlama kuralları
```

---

## Kurulum ve Kullanım

### 1. Bağımlılıkların Yüklenmesi

Gerekli Python paketlerini yüklemek için terminalde aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

### 2. Web Komuta Kontrol Panelinin (Web C2) Başlatılması

Arayüzü başlatmak için sunucu betiğini çalıştırın:

```bash
python web/server.py
```

Sunucu başladıktan sonra web tarayıcınızda aşağıdaki adrese gidin:

```
http://localhost:8000
```

Arayüz Özellikleri:
- Mod Seçimi: File Replayer (Arşiv Videoları) veya Live Web Stream (Canlı Yayın).
- Model Seçimi: YOLOv8x, YOLOv8l, YOLOv8m, YOLOv8s, YOLOv8n ve YOLO11x modelleri arasında dinamik geçiş.
- Deniz Haritası: Sağ üst köşedeki katman butonundan OpenSeaMap katmanı aktif edilebilir.
- Canlı Kalibrasyon: Kamera açısı, eğimi, yüksekliği ve görüş alanı anlık olarak ayarlanabilir.

### 3. Ground Truth Benchmark Değerlendirme Testi

Sistemin doğruluk metriklerini (MOTA, IDF1, Precision, Recall) Ground Truth verileriyle kıyaslamak için:

```bash
python tests/test_benchmark.py
```

---

## Veri Akış Modları

1. **File Mode (Çevrimdışı / Arşiv Simülasyonu):**
   Diskte yer alan MP4 videolarını ve zamana göre sıralanmış AIS CSV dosyalarını kare kare okuyarak gerçek zamanlı gibi işler. Başarım ölçümü ve testler için kullanılır.

2. **Live Web Stream Mode (Canlı İnternet Akışı):**
   `streamlink` kütüphanesi aracılığıyla YouTube üzerindeki 7/24 deniz canlı yayınlarından video akışı alır ve `aisstream.io` WebSocket servisine bağlanarak anlık coğrafi bölgedeki gerçek gemi telsiz verilerini çekip eşleştirir.