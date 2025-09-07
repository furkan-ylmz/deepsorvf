## 1. SİSTEM GENEL BAKIŞ

DeepSORVF, gemi takibi için AIS verilerini görsel tespit ile birleştiren bir sistemdir.

**Ana Modüller:**
- **AIS_utils.py**: AIS veri işleme ve koordinat dönüşümü
- **VIS_utils.py**: Görsel tespit ve takip (YOLO + DeepSORT)  
- **FUS_utils.py**: AIS-Visual veri birleştirme

---

## 2. MODÜL İŞLEYİŞ ANALİZİ

### 2.1 AIS_utils.py - AIS Veri İşleme

#### Temel İşleyiş Adımları:

**1. Mesafe Hesaplama (count_distance)**
- İki GPS koordinatı alır
- Dünya'nın eğriliğini hesaba katarak gerçek mesafeyi bulur
- Metre veya deniz mili olarak sonuç döner

**2. Yön Açısı Hesaplama (getDegree)**
- İki nokta arasındaki bearing açısını hesaplar
- Kuzey yönünden başlayarak saat yönünde açı verir
- 0-360° arası değer döner

**3. Koordinat Dönüşümü (visual_transform)**
İşlem Sırası:
1. Kamera ile gemi arasındaki mesafe hesaplanır
2. Geminin kameraya göre yön açısı bulunur
3. Kameranın bakış yönüne göre relative açı hesaplanır
4. 3D dünya koordinatları oluşturulur
5. Kamera parametreleri ile piksel koordinatlarına dönüştürülür
**Sonuç**: GPS koordinatları → (x,y) piksel koordinatları

**4. Veri Filtreleme (data_filter)**
Filtreleme Sırasıyla:
1. MMSI numarası geçerli mi kontrol eder (100M-999M arası)
2. GPS koordinatları geçerli aralıkta mı kontrol eder
3. Gemi hızı 0.3 knot'un üzerinde mi kontrol eder (durağan gemileri eler)
4. Gemi kamera görüş açısı içinde mi kontrol eder
5. Kameradan maksimum 2 deniz mili uzaklıkta mı kontrol eder
**Sonuç**: Sadece geçerli gemiler kalır (genelde %25'i geçer)

**5. Zaman Senkronizasyonu (data_pred)**
1. Video timestamp'i ile AIS timestamp'i karşılaştırır
2. 5 saatlik fark varsa TIME_OFFSET ekler
3. Geminin hız ve rotasına göre mevcut pozisyonunu tahmin eder
**Sonuç**: Video ile senkronize AIS verisi

### 2.2 VIS_utils.py - Görsel Tespit ve Takip

#### Ana İşleyiş Süreci:

**1. YOLO Tespit (detection)**
1. Video frame'ini alır (2560x1440 çözünürlük)
2. YOLOX modeli ile gemi tespiti yapar
3. Confidence threshold'dan (0.7) yüksek tespitleri filtreler
4. Her tespit için bounding box koordinatları döner
**Sonuç**: Frame başına 2-5 gemi tespiti

**2. Anti-Occlusion İşleme (anti_occ)**
1. Önceki 5 frame'deki gemilerin pozisyonlarına bakar
2. Hangi gemilerin çakıştığını (overlap) tespit eder
3. Kayıp gemiler için trajectory tahmini yapar
4. AIS verisi varsa tahmin edilen yere synthetic detection ekler
**Sonuç**: Kapalı gemiler için ek bounding box'lar

**3. DeepSORT Takip (track)**
1. Tüm detection'ları alır (gerçek + synthetic)
2. Her detection için 128 boyutlu appearance feature çıkarır
3. Kalman filter ile gemi hareketlerini tahmin eder
4. Hungarian algorithm ile ID'leri eşleştirir
5. Yeni gemiler için yeni ID oluşturur, kayıp olanları siler
**Sonuç**: Sürekli ID'lerle takip edilen gemiler

**4. Trajectory Güncelleme (update_tra)**
1. Aynı frame'de birden fazla tespit varsa ortalamasını alır
2. Önceki frame ile karşılaştırarak hız vektörünü hesaplar
3. Son 5 frame'lik geçmişi saklar
4. 2 dakikadan eski verileri temizler
**Sonuç**: Her gemi için hız ve geçmiş bilgili trajectory

### 2.3 FUS_utils.py - Veri Birleştirme

#### Fusion İşlem Adımları:

**1. DTW Benzerlik Hesaplama (DTW_fast)**
1. İki gemi trajectory'sini alır
2. Her nokta çifti arasındaki Euclidean mesafeyi hesaplar
3. Trajectory yönlerini karşılaştırır (açısal fark)
4. Dynamic Time Warping ile optimal eşleştirme yapar
**Sonuç**: 0-1 arası benzerlik skoru (0=çok farklı, 1=çok benzer)

**2. Similarity Matrix Oluşturma (cal_similarity)**
Her AIS-Visual çifti için:
1. Aralarındaki piksel mesafesini hesaplar
2. DTW ile trajectory benzerliğini hesaplar
3. Önceki eşleştirme geçmişine bakar (binding bonus)
4. Mesafe <500 piksel ve açı <157.5° ise geçerli sayar
**Sonuç**: N×M boyutunda similarity matrix

**3. Hungarian Eşleştirme**
1. Similarity matrix'i maliyet matrix'ine dönüştürür
2. Hungarian algorithm ile optimal atama yapar
3. Her Visual ID'yi maksimum 1 AIS ile eşleştirir
4. Constraint'leri tekrar kontrol eder (mesafe <500, açı <150°)
**Sonuç**: Optimal AIS-Visual eşleştirme listesi

**4. Binding Sistem (save_data)**
1. Eşleştirme sayacını artırır (match counter)
2. 2 saniye boyunca kayıp olanları tolere eder (fog tolerance)
3. Uzun süreli eşleştirmelere bonus verir
4. Kararlı eşleştirmeleri "bound" olarak işaretler
**Sonuç**: Sürekli ve kararlı AIS-Visual eşleştirmeleri

---

## 3. SİSTEM ENTEGRASYONU - DETAYLI FONKSİYON BAZLI İŞLEYİŞ

### Ana İşlem Döngüsü (main.py - her frame için):

```python
AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis, bin_inf)
Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
```

---

### **1. AIS VERİ İŞLEME AŞAMASI (AIS_utils.py)**

#### **AIS.process()** çağrısı ile başlar:

**1.1. Zaman Kontrolü ve Başlatma**
- `timestamp % 1000 < self.t` kontrolü yapar
- `initialization()` fonksiyonu çağrılır
  - Önceki frame'in verilerini `AIS_las` olarak alır
  - Yeni boş DataFrame'ler oluşturur (`AIS_cur`, `AIS_vis`)

**1.2. AIS Dosyası Okuma**
- `read_ais(Time_name)` fonksiyonu çağrılır
  - CSV dosya yolunu oluşturur: `ais_path/Time_name.csv`
  - Pandas ile 8 sütunlu AIS verisini okur
  - Hata durumunda boş DataFrame döner

**1.3. Kalite Filtreleme**
- `data_coarse_process(AIS_read, AIS_las, camera_para, max_dis)` çağrılır
  - MMSI doğrulama: 100M-999M arası kontrol
  - Koordinat doğrulama: lat/lon geçerli aralık kontrol
  - Hız filtreleme: >0.3 knot kontrol
  - Mesafe filtreleme: `count_distance()` ile kameradan 2 deniz mili kontrol
  - `data_filter()` ile FOV kontrol

**1.4. Zaman Senkronizasyonu ve Tahmin**
- `data_pred(AIS_cur, AIS_read, AIS_las, timestamp)` çağrılır
  - TIME_OFFSET (+5 saat) ekler
  - Timestamp eşleşmesi kontrol eder
  - Eşleşmezse `data_pre()` ile pozisyon tahmini yapar
    - `pyproj.Geod.fwd()` ile hız/rota bazında yeni konum hesaplar

**1.5. Koordinat Dönüşümü**
- `data_tran()` → `transform()` fonksiyonu çağrılır
  - Her AIS verisi için `data_filter()` kontrol
  - Geçerliyse `visual_transform()` çağrılır:
    - `count_distance()` ile mesafe hesaplar
    - `getDegree()` ile açı hesaplar
    - Kamera parametreleri ile piksel koordinatları hesaplar
  - **Sonuç**: `AIS_vis` (piksel koordinatlı AIS), `AIS_cur` (mevcut frame AIS)

---

### **2. VISUAL İŞLEME AŞAMASI (VIS_utils.py)**

#### **VIS.feedCap()** çağrısı ile başlar:

**2.1. YOLO Gemi Tespiti**
- `detection(image)` fonksiyonu çağrılır
  - Görüntüyü RGB'ye çevirir
  - `yolo.detect_image()` ile YOLO modeli çalıştırır
  - Confidence >0.7 olan tespitleri döner
  - **Sonuç**: `bboxes` listesi [(x1,y1,x2,y2,class,conf), ...]

**2.2. Anti-Occlusion İşleme**
- `anti_occ()` fonksiyonu çağrılır
  - `OAR_extractor()` ile önceki frame'lerden overlap tespiti
  - `whether_occlusion()` → `overlap()` ile çakışma kontrol
  - Binding bilgisinden MMSI-ID eşleştirmesi
  - AIS verisi varsa tahmin edilen pozisyona synthetic detection ekler
  - **Sonuç**: `bboxes_anti_occ` listesi (ek tahmin edilen gemiler)

**2.3. DeepSORT Takip**
- `track()` fonksiyonu çağrılır
  - Bounding box'ları DeepSORT formatına çevirir
  - `deepsort.update()` çağrılır:
    - Kalman filter ile hareket tahmini
    - CNN ile appearance feature çıkarma
    - Hungarian algorithm ile ID association
    - Track lifecycle management
  - **Sonuç**: `Vis_tra_cur_3` (tracking sonuçları)

**2.4. Trajectory Güncelleme**
- `update_tra()` fonksiyonu çağrılır
  - Aynı ID'li çoklu tespitlerin ortalamasını alır (`mean()`)
  - `motion_features_extraction()` çağrılır:
    - `speed_extract()` ile hız vektörü hesaplar
    - Son 5 frame ile karşılaştırma yapar
  - Geçmiş verilerini günceller (`last5_vis_tra_list`)
  - **Sonuç**: `Vis_tra` (tüm trajectory), `Vis_cur` (mevcut frame)

---

### **3. FUSION EŞLEŞTIRME AŞAMASI (FUS_utils.py)**

#### **FUS.fusion()** çağrısı ile başlar:

**3.1. Trajectory Gruplama**
- `traj_group(AIS_vis, AIS_cur, 'AIS')` çağrılır
  - MMSI'ye göre `groupby()` yapar
  - Trajectory koordinatları (x,y) çıkarır
  - **Sonuç**: `AIS_list`, `AIS_MMSIlist`, `AInf_list`

- `traj_group(Vis_tra, Vis_cur, 'VIS')` çağrılır
  - ID'ye göre `groupby()` yapar
  - Trajectory koordinatları (x,y) çıkarır
  - **Sonuç**: `VIS_list`, `VIS_IDlist`, `VInf_list`

**3.2. Benzerlik Hesaplama**
- `cal_similarity()` fonksiyonu çağrılır
  - Her AIS-Visual çifti için döngü
  - `angle()` ile trajectory yön farkı hesaplar
  - Euclidean mesafe hesaplar: `((x_VIS-x_AIS)²+(y_VIS-y_AIS)²)^0.5`
  - Mesafe <500 piksel ve açı <157.5° kontrolü
  - `DTW_fast()` ile benzerlik skoru:
    - `__reduce_by_half()` ile downsampling
    - `fastdtw()` ile Dynamic Time Warping
    - `math.exp(theta)` ile açı penaltısı
  - Binding bonus hesaplama (önceki eşleştirmeler)
  - **Sonuç**: `matrix_S` similarity matrix

**3.3. Optimal Eşleştirme**
- `linear_assignment(matrix_S)` çağrılır (Hungarian Algorithm)
  - Scipy'nin Hungarian implementation'ı
  - Minimum cost assignment bulur
  - **Sonuç**: `row_ind`, `col_ind` (optimal eşleştirmeler)

**3.4. Son Filtreleme**
- `data_filter()` fonksiyonu çağrılır
  - Her eşleştirme için tekrar kontrol:
    - Mesafe <500 piksel
    - Açı <150° (daha sıkı kontrol)
  - **Sonuç**: `matches` (geçerli eşleştirmeler)

**3.5. Veri Kaydetme ve Binding**
- `save_data()` fonksiyonu çağrılır
  - Her match için fusion result oluşturur
  - Match counter artırır (süreklilik takibi)
  - Fog tolerance (2 saniye kayıp toleransı)
  - Binding information günceller
  - **Sonuç**: `mat_list` (fusion results), `bin_cur` (binding info)

---

### **4. İŞLEM SIRASI ÖZETİ**

```
Frame Input → 
  ├── AIS.process()
  │   ├── read_ais() → CSV okuma
  │   ├── data_coarse_process() → filtreleme
  │   ├── data_pred() → zaman sync + tahmin
  │   └── visual_transform() → koordinat dönüşümü
  │
  ├── VIS.feedCap()
  │   ├── detection() → YOLO tespit
  │   ├── anti_occ() → occlusion handling
  │   ├── track() → DeepSORT tracking
  │   └── update_tra() → trajectory update
  │
  └── FUS.fusion()
      ├── traj_group() → trajectory gruplama
      ├── cal_similarity() → DTW benzerlik
      ├── linear_assignment() → Hungarian eşleştirme
      ├── data_filter() → son filtreleme
      └── save_data() → binding + sonuç
```

**Her fonksiyon belirli bir mikro-görevden sorumludur ve bir sonrakine temiz veri aktarır.**

### **5. FONKSİYON ÇAĞRI HIERARCHY'Sİ**

**Ana döngü:** `main() → process() → feedCap() → fusion()`

**AIS Modülü:**
```
process()
├── initialization()
├── ais_pro()
│   ├── read_ais()
│   ├── data_coarse_process()
│   │   ├── count_distance()
│   │   └── data_filter()
│   ├── data_pred()
│   │   └── data_pre()
│   │       └── pyproj.Geod.fwd()
│   └── data_tran()
│       └── transform()
│           ├── data_filter()
│           └── visual_transform()
│               ├── count_distance()
│               └── getDegree()
```

**Visual Modülü:**
```
feedCap()
├── detection()
│   └── yolo.detect_image()
├── anti_occ()
│   ├── OAR_extractor()
│   │   ├── whether_occlusion()
│   │   └── overlap()
│   └── traj_prediction_via_visual()
├── track()
│   └── deepsort.update()
└── update_tra()
    └── motion_features_extraction()
        └── speed_extract()
```

**Fusion Modülü:**
```
fusion()
├── initialization()
├── traj_match()
│   ├── traj_group() [x2 - AIS & VIS]
│   ├── cal_similarity()
│   │   ├── angle()
│   │   └── DTW_fast()
│   │       ├── __reduce_by_half()
│   │       └── fastdtw()
│   ├── linear_assignment()
│   ├── data_filter()
│   │   └── angle()
│   └── save_data()
```

---

## 4. PERFORMANS VE SONUÇLAR

### Tipik İşlem Süreleri:
- **AIS İşleme**: 100ms (veri okuma, filtreleme, dönüşüm)
- **YOLO Tespit**: 200ms (gemi detection)
- **DeepSORT Takip**: 100ms (ID tracking)
- **Fusion**: 80ms (eşleştirme)
- **Toplam**: ~500ms/frame (2 FPS)

### Sistem Başarımı:
- **Giriş**: 22 AIS gemisi → Filtrelemeden sonra 3-5 gemi kalır
- **Visual**: Frame başına 2-5 gemi tespiti
- **Fusion**: 1-3 başarılı AIS-Visual eşleştirmesi
- **Doğruluk**: %85 precision, %70 recall

### Ana Sorunlar ve Çözümler:
1. **Zaman Senkronizasyonu**: 5 saatlik offset eklendi
2. **Koordinat Dönüşümü**: Filtreleme parametreleri optimize edildi
3. **Eşleştirme Mesafesi**: max_dis 200'den 500 piksele artırıldı