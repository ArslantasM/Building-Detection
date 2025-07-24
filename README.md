# SpaceNet Bina Tespiti - YOLOv8 ONNX (Python)

Bu proje, SpaceNet veri setini kullanarak bina tespiti için YOLOv8 modelini eğitir ve ONNX formatına dönüştürür.

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

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. AWS CLI kurulumunu kontrol edin:
```bash
aws --version
```

## Kullanım

1. Python betiğini çalıştırın:
```bash
python spacenet_buildingdetection_yolov8_onnx.py
```

## İşlem Adımları

1. **Veri Seti İndirme**
   - SpaceNet veri seti AWS'den indirilir
   - İndirilen arşiv otomatik olarak açılır
   - Eğer veri seti zaten mevcutsa, bu adım atlanır

2. **Görüntü İşleme**
   - TIFF görüntüler JPG formatına dönüştürülür
   - Görüntüler normalize edilir ve iyileştirilir
   - İşlem durumu progress bar ile gösterilir

3. **Etiket Dönüştürme**
   - GeoJSON formatındaki etiketler YOLO formatına dönüştürülür
   - Dönüştürme işlemi progress bar ile takip edilir

4. **Veri Seti Hazırlama**
   - Görüntüler eğitim ve doğrulama setlerine ayrılır (%80 eğitim, %20 doğrulama)
   - Dosyalar ilgili dizinlere kopyalanır

5. **Model Eğitimi**
   - YOLOv8n modeli eğitilir
   - Eğitim parametreleri:
     - Epochs: 50
     - Batch size: 16
     - Görüntü boyutu: 640x640
     - Optimizer: AdamW
     - Learning rate: 0.001

6. **ONNX Dönüşümü**
   - Eğitilen model ONNX formatına dönüştürülür
   - ONNX model test edilir

## Çıktılar

- `spacenet.yaml`: YOLOv8 yapılandırma dosyası
- `yolov8n.onnx`: ONNX formatında model
- `runs/detect/spacenet_buildings/`: Eğitim sonuçları ve grafikler

## Notlar

- Veri seti büyük olduğu için indirme işlemi uzun sürebilir
- GPU kullanımı önerilir
- İşlemler sırasında ilerleme durumu ve istatistikler gösterilir
- Hata durumunda detaylı bilgi verilir 
