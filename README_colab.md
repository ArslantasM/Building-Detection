# SpaceNet Bina Tespiti - YOLOv8 ONNX (Colab)

Bu notebook, SpaceNet veri setini kullanarak bina tespiti için YOLOv8 modelini eğitir ve ONNX formatına dönüştürür.

## Özellikler

- Tek kod bloğunda tüm işlemler
- Progress bar ile işlem takibi
- Otomatik veri seti indirme ve hazırlama
- Google Drive entegrasyonu
- Hata yönetimi ve detaylı loglama

## Kullanım

1. Notebook'u Google Colab'da açın
2. Gerekli izinleri verin (Google Drive erişimi)
3. Kod bloğunu çalıştırın

## İşlem Adımları

1. **Kurulum**
   - Gerekli paketler yüklenir
   - GPU kontrolü yapılır
   - Google Drive bağlanır

2. **Veri Seti İndirme**
   - SpaceNet veri seti AWS'den indirilir
   - İndirilen arşiv otomatik olarak açılır
   - Eğer veri seti zaten mevcutsa, bu adım atlanır

3. **Görüntü İşleme**
   - TIFF görüntüler JPG formatına dönüştürülür
   - Görüntüler normalize edilir ve iyileştirilir
   - İşlem durumu progress bar ile gösterilir

4. **Etiket Dönüştürme**
   - GeoJSON formatındaki etiketler YOLO formatına dönüştürülür
   - Dönüştürme işlemi progress bar ile takip edilir

5. **Veri Seti Hazırlama**
   - Görüntüler eğitim ve doğrulama setlerine ayrılır (%80 eğitim, %20 doğrulama)
   - Dosyalar ilgili dizinlere kopyalanır

6. **Model Eğitimi**
   - YOLOv8n modeli eğitilir
   - Eğitim parametreleri:
     - Epochs: 50
     - Batch size: 16
     - Görüntü boyutu: 640x640
     - Optimizer: AdamW
     - Learning rate: 0.001

7. **ONNX Dönüşümü**
   - Eğitilen model ONNX formatına dönüştürülür
   - ONNX model test edilir

8. **Sonuçları Kaydetme**
   - Eğitim sonuçları Google Drive'a kaydedilir
   - ONNX model Google Drive'a kaydedilir

## Çıktılar

- `spacenet.yaml`: YOLOv8 yapılandırma dosyası
- `yolov8n.onnx`: ONNX formatında model
- `runs/detect/spacenet_buildings/`: Eğitim sonuçları ve grafikler

## Gereksinimler

```bash
docutils>=0.20,<0.22
ultralytics
geopandas
shapely
onnxruntime
rasterio
scikit-learn
opencv-python
awscli
tqdm
```

## Notlar

- Veri seti büyük olduğu için indirme işlemi uzun sürebilir
- Colab Pro önerilir (daha fazla RAM ve GPU süresi)
- İşlemler sırasında ilerleme durumu ve istatistikler gösterilir
- Hata durumunda detaylı bilgi verilir
- Google Drive'da yeterli boş alan olduğundan emin olun 
