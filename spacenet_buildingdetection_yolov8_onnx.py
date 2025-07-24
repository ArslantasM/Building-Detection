import os
import shutil
import subprocess
import sys
from PIL import Image
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import geopandas as gpd
from shapely.geometry import box
from sklearn.model_selection import train_test_split
import cv2
import warnings
import tqdm
warnings.filterwarnings('ignore')

def check_dataset_status():
    """Veri seti durumunu kontrol et"""
    spacenet_file = "SN2_buildings_train_AOI_2_Vegas.tar.gz"
    extract_dir = "./spacenet"
    
    print("\nDosya ve Klasör Durumu Kontrol Ediliyor...")
    print("-" * 40)
    
    # Arşiv dosyası kontrolü
    if os.path.exists(spacenet_file):
        file_size = os.path.getsize(spacenet_file) / (1024 * 1024 * 1024)  # GB cinsinden
        print(f"✓ Arşiv dosyası mevcut: {spacenet_file}")
        print(f"  └─ Boyut: {file_size:.2f} GB")
    else:
        print(f"✗ Arşiv dosyası bulunamadı: {spacenet_file}")
    
    # Ana klasör kontrolü
    if os.path.exists(extract_dir):
        print(f"✓ Ana klasör mevcut: {extract_dir}")
        
        # Alt klasör kontrolleri
        subfolders = [
            "AOI_2_Vegas_Train",
            "AOI_2_Vegas_Train/RGB-PanSharpen",
            "AOI_2_Vegas_Train/geojson/buildings"
        ]
        
        all_subfolders_exist = True
        for subfolder in subfolders:
            full_path = os.path.join(extract_dir, subfolder)
            if os.path.exists(full_path):
                file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
                print(f"  ├─ ✓ Alt klasör mevcut: {subfolder}")
                print(f"  │   └─ İçerik: {file_count} dosya")
            else:
                print(f"  ├─ ✗ Alt klasör eksik: {subfolder}")
                all_subfolders_exist = False
        
        if all_subfolders_exist:
            print("  └─ Tüm gerekli alt klasörler mevcut ve dolu")
        else:
            print("  └─ Bazı alt klasörler eksik veya boş!")
    else:
        print(f"✗ Ana klasör bulunamadı: {extract_dir}")
    
    print("\nİşlem önerisi:")
    if not os.path.exists(spacenet_file):
        print("- Arşiv dosyası eksik. Lütfen veri setini indirin.")
        return False
    elif not os.path.exists(extract_dir) or not all_subfolders_exist:
        print("- Arşiv dosyası mevcut ama açılmamış. Arşiv açılacak.")
        return True
    else:
        print("- Tüm dosya ve klasörler hazır. İşleme devam edilebilir.")
        return True

def normalize_image(img):
    """Görüntüyü normalize et ve kontrastı artır"""
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(img.shape) == 3:
        for i in range(3):
            img[:,:,i] = clahe.apply(img[:,:,i].astype(np.uint8))
    else:
        img = clahe.apply(img.astype(np.uint8))
    return img

def resize_image(img, target_size=(640, 640)):
    """Görüntüyü hedef boyuta yeniden boyutlandır"""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def enhance_edges(img):
    """Görüntüdeki kenarları belirginleştir"""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def geojson_to_yolo(geojson_path, image_size, out_path):
    """GeoJSON'u YOLO formatına dönüştür"""
    try:
        gdf = gpd.read_file(geojson_path)
        with open(out_path, 'w') as f:
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == "Polygon":
                    minx, miny, maxx, maxy = geom.bounds
                    x_center = ((minx + maxx) / 2) / image_size[0]
                    y_center = ((miny + maxy) / 2) / image_size[1]
                    width = (maxx - minx) / image_size[0]
                    height = (maxy - miny) / image_size[1]
                    
                    x_center = max(0.005, min(0.995, x_center))
                    y_center = max(0.005, min(0.995, y_center))
                    width = max(0.01, min(0.99, width))
                    height = max(0.01, min(0.99, height))
                    
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        return True
    except Exception as e:
        print(f"Uyarı: {os.path.basename(geojson_path)} dönüştürülemedi: {e}")
        return False

# Veri seti durumunu kontrol et
if not check_dataset_status():
    print("\nLütfen veri setini indirip tekrar deneyin.")
    sys.exit(1)

# SpaceNet veri setini hazırla
spacenet_file = "SN2_buildings_train_AOI_2_Vegas.tar.gz"
extract_dir = "./spacenet"

# Eğer veri seti klasörü yoksa veya eksik/boşsa, arşivi aç
if not os.path.exists(extract_dir) or not os.path.exists(os.path.join(extract_dir, "AOI_2_Vegas_Train")):
    print("\nVeri seti arşivden çıkartılıyor...")
    if not os.path.exists("spacenet"):
        os.makedirs("spacenet")
    subprocess.run(["tar", "-xf", spacenet_file, "-C", "spacenet"], check=True)
    print("Veri seti hazır!")

# Görüntüleri işle
src_dir = "./spacenet/AOI_2_Vegas_Train/RGB-PanSharpen"
dst_dir = "./spacenet/RGB-Converted"
label_dir = "./spacenet/labels"

os.makedirs(dst_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Görüntüleri dönüştür
files = [f for f in os.listdir(src_dir) if f.endswith('.tif') and "AOI_2_Vegas_img" in f]
converted = 0
skipped = 0

print("\nGörüntüler işleniyor...")
for file in tqdm.tqdm(files, desc="Görüntü İşleme"):
    img_id = file.split("RGB-PanSharpen_")[-1].replace(".tif", "")
    geojson_path = os.path.join("./spacenet/AOI_2_Vegas_Train/geojson/buildings", f"buildings_{img_id}.geojson")
    
    if not os.path.exists(geojson_path):
        skipped += 1
        continue

    src_path = os.path.join(src_dir, file)
    try:
        with rasterio.open(src_path) as src:
            rgb = src.read([1, 2, 3])
            img = reshape_as_image(rgb)
            img = normalize_image(img)
            img = enhance_edges(img)
            img = resize_image(img)
            
            out_path = os.path.join(dst_dir, f"{img_id}.jpg")
            Image.fromarray(img.astype(np.uint8)).save(out_path, quality=95)
            converted += 1
    except Exception as e:
        print(f"\nUyarı: {file} dönüştürülemedi: {e}")
        skipped += 1

print(f"\nGörüntü Dönüştürme İstatistikleri:")
print(f"Başarılı: {converted} görüntü")
print(f"Atlanan: {skipped} görüntü")

# Etiketleri dönüştür
image_size = None
for img_file in os.listdir(dst_dir):
    if img_file.endswith('.jpg'):
        with Image.open(os.path.join(dst_dir, img_file)) as img:
            image_size = img.size
            print(f"\nReferans görüntü boyutu: {image_size}")
            break

if image_size:
    converted_labels = 0
    failed_labels = 0
    
    print("\nEtiketler dönüştürülüyor...")
    for img_file in tqdm.tqdm(os.listdir(dst_dir), desc="Etiket Dönüştürme"):
        if img_file.endswith('.jpg'):
            img_id = img_file.replace('.jpg', '')
            geojson_path = os.path.join("./spacenet/AOI_2_Vegas_Train/geojson/buildings", f"buildings_{img_id}.geojson")
            label_path = os.path.join(label_dir, f"{img_id}.txt")
            
            if geojson_to_yolo(geojson_path, image_size, label_path):
                converted_labels += 1
            else:
                failed_labels += 1

    print(f"\nEtiket Dönüştürme İstatistikleri:")
    print(f"Başarılı: {converted_labels} etiket")
    print(f"Başarısız: {failed_labels} etiket")

# Veri setini hazırla
all_images = []
for img_file in os.listdir(dst_dir):
    if img_file.endswith('.jpg'):
        img_id = img_file.replace('.jpg', '')
        label_path = os.path.join(label_dir, f"{img_id}.txt")
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            all_images.append(img_file)

if len(all_images) > 0:
    # Eğitim/doğrulama ayrımı
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Dizinleri oluştur
    train_img_dir = "./spacenet/images/train"
    val_img_dir = "./spacenet/images/val"
    train_label_dir = "./spacenet/labels/train"
    val_label_dir = "./spacenet/labels/val"
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Dosyaları kopyala
    def copy_files(images, img_dir, label_dir_src, label_dir_dst):
        copied = 0
        print(f"\nDosyalar kopyalanıyor...")
        for img_file in tqdm.tqdm(images, desc="Dosya Kopyalama"):
            try:
                img_id = img_file.replace('.jpg', '')
                shutil.copy(
                    os.path.join(dst_dir, img_file),
                    os.path.join(img_dir, img_file)
                )
                shutil.copy(
                    os.path.join(label_dir_src, f"{img_id}.txt"),
                    os.path.join(label_dir_dst, f"{img_id}.txt")
                )
                copied += 1
            except Exception as e:
                print(f"\nUyarı: {img_file} kopyalanamadı: {e}")
        return copied
    
    train_copied = copy_files(train_imgs, train_img_dir, label_dir, train_label_dir)
    val_copied = copy_files(val_imgs, val_img_dir, label_dir, val_label_dir)
    
    print(f"\nVeri Seti Ayrım İstatistikleri:")
    print(f"Eğitim: {train_copied} görüntü")
    print(f"Doğrulama: {val_copied} görüntü")

# YOLOv8 yapılandırma dosyasını oluştur
yaml_content = f"""
# YOLOv8 yapılandırması
path: {os.path.abspath("./spacenet")}  # veri seti kök dizini
train: images/train  # eğitim görüntüleri
val: images/val  # doğrulama görüntüleri

# Sınıflar
nc: 1  # sınıf sayısı
names: ['building']  # sınıf isimleri

# Eğitim parametreleri
epochs: 50
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
fl_gamma: 0.0
label_smoothing: 0.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
"""

with open("spacenet.yaml", "w") as f:
    f.write(yaml_content)
print("spacenet.yaml başarıyla oluşturuldu.")

# YOLOv8 modelini eğit
from ultralytics import YOLO

print("\nModel Eğitim Parametreleri:")
print("  - Model: YOLOv8n")
print("  - Epochs: 50")
print("  - Görüntü boyutu: 640x640")
print("  - Batch size: 16")
print("  - Optimizer: AdamW")
print("  - Learning rate: 0.001")

# Yeni model oluştur
model = YOLO('yolov8n.yaml')  # YOLOv8n modelini sıfırdan oluştur
model.train(
    data="spacenet.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name='spacenet_buildings',
    optimizer='AdamW',
    lr0=0.001
)

# ONNX'e dönüştür ve test et
print("\nModel ONNX formatına dönüştürülüyor...")
model.export(format="onnx")
print("ONNX model oluşturuldu: runs/detect/spacenet_buildings/weights/best.onnx")

print("\nONNX model test ediliyor...")
import onnxruntime
session = onnxruntime.InferenceSession("runs/detect/spacenet_buildings/weights/best.onnx")
input_name = session.get_inputs()[0].name
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
output = session.run(None, {input_name: dummy_input})
print("ONNX model testi başarılı!")

# Sonuçları kaydet
print("\nSonuçlar kaydediliyor...")
if not os.path.exists("results"):
    os.makedirs("results")
shutil.copytree("runs/detect/spacenet_buildings", "results/spacenet_buildings", dirs_exist_ok=True)
shutil.copy("runs/detect/spacenet_buildings/weights/best.onnx", "results/")
print("Sonuçlar başarıyla kaydedildi!")