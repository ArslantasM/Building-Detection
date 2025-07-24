# SpaceNet Bina Tespiti - YOLOv8 ONNX (Colab)

Bu Colab notebook'u, SpaceNet veri setini kullanarak bina tespiti için YOLOv8 modelini eğitir ve ONNX formatına dönüştürür. Google Colab'in sunduğu ücretsiz GPU ve depolama kaynaklarını kullanır.

## Gereksinimler

- Google hesabı
- Google Drive'da en az 50GB boş alan
- İnternet bağlantısı

## Notebook Yapısı

1. **Kurulum**
   - GPU kontrolü
   - Google Drive bağlantısı
   - Gerekli paketlerin yüklenmesi

2. **Veri İndirme**
   - AWS CLI kurulumu
   - SpaceNet veri seti indirme
   - Arşiv çıkarma

3. **Görüntü İşleme**
   - Kontrast iyileştirme
   - Kenar belirginleştirme
   - Boyut düzenleme
   - Format dönüşümü

4. **Etiket Dönüşümü**
   - GeoJSON -> YOLO formatı
   - Koordinat normalizasyonu
   - Etiket doğrulama

5. **Veri Seti Hazırlama**
   - Eğitim/doğrulama bölünmesi
   - Dizin yapısı oluşturma
   - Dosya kopyalama

6. **Model Eğitimi**
   - YOLOv8 yapılandırması
   - Eğitim başlatma
   - ONNX dönüşümü

7. **Sonuç Kaydetme**
   - Drive'a model kaydetme
   - Eğitim sonuçlarını saklama

## Kullanım

1. Notebook'u Colab'de açın:
   ```
   File > Open notebook > GitHub > [URL'yi yapıştırın]
   ```

2. GPU'yu etkinleştirin:
   ```
   Runtime > Change runtime type > GPU
   ```

3. Google Drive'a bağlanın:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Hücreleri sırayla çalıştırın

## Eğitim Parametreleri

- Model: YOLOv8n
- Epochs: 50
- Batch Size: 16
- Görüntü Boyutu: 640x640
- Optimizer: AdamW
- Learning Rate: 0.001

## Çıktılar

Google Drive'da oluşturulan dosyalar:
- `/content/drive/MyDrive/spacenet_buildings/`: Eğitim sonuçları
- `/content/drive/MyDrive/yolov8n.onnx`: ONNX model

## Performans

- Eğitim süresi: ~4-6 saat (Colab T4 GPU)
- RAM kullanımı: ~12GB
- Disk kullanımı: ~50GB

## Sınırlamalar

1. **Colab Limitleri**:
   - 12 saat maksimum çalışma süresi
   - Belirli süre sonra GPU bağlantısı kopabilir
   - Sınırlı RAM ve disk alanı

2. **Veri Seti**:
   - İndirme süresi uzun olabilir
   - Disk alanı yetmeyebilir
   - Drive'a kaydetme zaman alabilir

## İpuçları

1. **GPU Kullanımı**:
   - Uzun işlemler için Pro sürüme geçin
   - GPU bağlantısını kontrol edin
   - Kullanılmadığında runtime'ı durdurun

2. **Veri Yönetimi**:
   - Önemli dosyaları Drive'a kaydedin
   - Gereksiz dosyaları silin
   - Checkpoint'leri saklayın

3. **Hata Yönetimi**:
   - Hata mesajlarını okuyun
   - Runtime'ı yeniden başlatın
   - Gerekirse hücreleri tekrar çalıştırın

## Notlar

- Ücretsiz Colab GPU'su yeterlidir
- İnternet bağlantısı önemlidir
- Drive yedeği önerilir
- Uzun eğitimler için Pro sürüm düşünülebilir 