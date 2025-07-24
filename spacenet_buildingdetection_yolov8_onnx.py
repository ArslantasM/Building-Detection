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
from google.colab import drive
warnings.filterwarnings('ignore')

def check_gpu():
    """GPU durumunu kontrol et"""
    try:
        gpu_info = subprocess.check_output(['nvidia-smi']).decode()
        print("GPU Bilgisi:")
        print(gpu_info)
    except:
        print("GPU bulunamadı!")

def install_requirements():
    """Gerekli paketleri yükle"""
    print("Gerekli paketler yükleniyor...")
    packages = [
        "docutils>=0.20,<0.22",
        "ultralytics",
        "geopandas",
        "shapely",
        "onnxruntime",
        "rasterio",
        "scikit-learn",
        "opencv-python",
        "awscli",
        "tqdm"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("Paket kurulumu tamamlandı!")

def mount_drive():
    """Google Drive'ı bağla"""
    print("Google Drive bağlanıyor...")
    drive.mount('/content/drive')
    print("Google Drive bağlandı!")

def download_dataset():
    """SpaceNet veri setini indir"""
    spacenet_file = "SN2_buildings_train_AOI_2_Vegas.tar.gz"
    extract_dir = "./spacenet"
    
    # Eğer veri seti zaten indirilmiş ve açılmışsa atla
    if os.path.exists(extract_dir) and os.path.exists(os.path.join(extract_dir, "AOI_2_Vegas_Train")):
        print("Veri seti zaten mevcut, indirme ve açma işlemleri atlanıyor...")
        return
    
    print("SpaceNet veri seti indiriliyor...")
    if not os.path.exists("spacenet"):
        os.makedirs("spacenet")
    
    # Eğer arşiv dosyası zaten varsa indirme işlemini atla
    if not os.path.exists(spacenet_file):
        print("Veri seti arşivi indiriliyor...")
        subprocess.run([
            "aws", "s3", "cp",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
            ".",
            "--no-sign-request"
        ], check=True)
    else:
        print("Arşiv dosyası zaten mevcut, indirme işlemi atlanıyor...")
    
    # Arşivi çıkart
    print("Veri seti arşivden çıkartılıyor...")
    subprocess.run([
        "tar", "-xf", spacenet_file, "-C", "spacenet"
    ], check=True)
    
    print("Veri seti hazır!")

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

def process_images():
    """Görüntüleri işle ve etiketleri dönüştür"""
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

def prepare_dataset():
    """Veri setini eğitim ve doğrulama için hazırla"""
    dst_dir = "./spacenet/RGB-Converted"
    label_dir = "./spacenet/labels"

    # Etiketli görüntüleri listele
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

def create_yaml():
    """YOLOv8 yapılandırma dosyasını oluştur"""
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

def train_model():
    """YOLOv8 modelini eğit"""
    from ultralytics import YOLO

    print("\nModel Eğitim Parametreleri:")
    print("  - Model: YOLOv8n")
    print("  - Epochs: 50")
    print("  - Görüntü boyutu: 640x640")
    print("  - Batch size: 16")
    print("  - Optimizer: AdamW")
    print("  - Learning rate: 0.001")

    model = YOLO("yolov8n.pt")
    model.train(
        data="spacenet.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name='spacenet_buildings',
        optimizer='AdamW',
        lr0=0.001
    )
    return model

def export_onnx(model):
    """Modeli ONNX formatına dönüştür ve test et"""
    print("\nModel ONNX formatına dönüştürülüyor...")
    model.export(format="onnx")
    print("ONNX model oluşturuldu: yolov8n.onnx")

    print("\nONNX model test ediliyor...")
    import onnxruntime
    session = onnxruntime.InferenceSession("yolov8n.onnx")
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    output = session.run(None, {input_name: dummy_input})
    print("ONNX model testi başarılı!")

def save_to_drive():
    """Sonuçları Google Drive'a kaydet"""
    print("\nSonuçlar Google Drive'a kaydediliyor...")
    shutil.copytree("runs/detect/spacenet_buildings", "/content/drive/MyDrive/spacenet_buildings")
    shutil.copy("yolov8n.onnx", "/content/drive/MyDrive/")
    print("Sonuçlar Google Drive'a başarıyla kaydedildi!")

def main():
    """Ana fonksiyon"""
    print("SpaceNet Bina Tespiti - YOLOv8 ONNX")
    print("====================================")
    
    # Gerekli paketleri yükle
    install_requirements()
    
    # GPU kontrolü
    check_gpu()
    
    # Google Drive'ı bağla
    mount_drive()
    
    # Veri setini indir
    download_dataset()
    
    # Görüntüleri işle
    process_images()
    
    # Veri setini hazırla
    prepare_dataset()
    
    # YAML dosyasını oluştur
    create_yaml()
    
    # Modeli eğit
    model = train_model()
    
    # ONNX'e dönüştür
    export_onnx(model)
    
    # Drive'a kaydet
    save_to_drive()

if __name__ == "__main__":
    main()