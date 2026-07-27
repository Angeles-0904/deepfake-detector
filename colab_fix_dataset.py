"""
Celda auxiliar para Colab: Organiza el dataset correctamente.
Copia esta función y ejecútala ANTES del Paso 7.
"""
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def organize_kaggle_dataset_colab():
    """
    Organiza el dataset de Kaggle en Colab de forma directa.
    Copia las imágenes a DATA_DIR que el entrenamiento espera.
    """
    # Rutas en Colab
    PROJECT_DIR = Path('/content/drive/MyDrive/detectorIA')
    DATA_DIR = PROJECT_DIR / 'data'
    KAGGLE_DIR = Path('/kaggle/input/140k-real-and-fake-faces')
    
    print("=" * 60)
    print("  ORGANIZANDO DATASET PARA ENTRENAMIENTO")
    print("=" * 60)
    
    # Verificar que el dataset de Kaggle existe
    if not KAGGLE_DIR.exists():
        print(f"❌ No se encontró el dataset en: {KAGGLE_DIR}")
        print("   Ejecuta primero el Paso 6 (descargar dataset)")
        return False
    
    print(f"✅ Dataset Kaggle encontrado en: {KAGGLE_DIR}")
    
    # Buscar carpeta de imágenes
    images_dir = KAGGLE_DIR / 'real_vs_fake'
    if not images_dir.exists():
        print(f"❌ No se encontró la carpeta real_vs_fake en: {KAGGLE_DIR}")
        return False
    
    print(f"✅ Carpeta de imágenes: {images_dir}")
    
    # Verificar CSVs
    train_csv = KAGGLE_DIR / 'train.csv'
    test_csv = KAGGLE_DIR / 'test.csv'
    valid_csv = KAGGLE_DIR / 'valid.csv'
    
    if not all([train_csv.exists(), test_csv.exists(), valid_csv.exists()]):
        print("❌ No se encontraron los archivos CSV (train.csv, test.csv, valid.csv)")
        return False
    
    print("✅ Archivos CSV encontrados")
    
    # Crear carpetas destino
    for split in ['train', 'validation', 'test']:
        for cls in ['real', 'fake']:
            dest_dir = DATA_DIR / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapeo de splits
    splits = {
        'train': (train_csv, DATA_DIR / 'train'),
        'validation': (valid_csv, DATA_DIR / 'validation'),
        'test': (test_csv, DATA_DIR / 'test'),
    }
    
    total_copied = 0
    
    for split_name, (csv_file, dest_base) in splits.items():
        print(f"\n📂 Procesando {split_name}...")
        
        # Leer CSV
        df = pd.read_csv(csv_file)
        
        # Buscar columnas correctas
        path_col = None
        label_col = None
        
        for col in df.columns:
            if col.lower() == 'path':
                path_col = col
            elif col.lower() == 'label':
                label_col = col
        
        if path_col is None or label_col is None:
            print(f"  ❌ Columnas no encontradas. Columnas del CSV: {list(df.columns)}")
            continue
        
        print(f"  Columnas: path='{path_col}', label='{label_col}'")
        
        # Crear subcarpetas
        real_dir = dest_base / 'real'
        fake_dir = dest_base / 'fake'
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar imágenes
        real_count = 0
        fake_count = 0
        
        for _, row in tqdm(df.iterrows(), desc=f"  Copiando {split_name}", total=len(df)):
            img_path = str(row[path_col])  # Ej: "train/real/31355.jpg"
            label = int(row[label_col])    # 0 = fake, 1 = real
            
            # Construir ruta origen
            src = images_dir / img_path
            
            # Si no existe, intentar solo el nombre
            if not src.exists():
                src = images_dir / Path(img_path).name
            
            if src.exists():
                dest = real_dir / src.name if label == 1 else fake_dir / src.name
                
                if not dest.exists():
                    shutil.copy2(src, dest)
                
                if label == 1:
                    real_count += 1
                else:
                    fake_count += 1
        
        total_copied += real_count + fake_count
        print(f"  ✅ {split_name}: {real_count} reales, {fake_count} falsas")
    
    # Verificar resultado
    print("\n" + "=" * 60)
    print("  VERIFICACIÓN FINAL")
    print("=" * 60)
    
    all_ok = True
    for split in ['train', 'validation', 'test']:
        for cls in ['real', 'fake']:
            count = len(list((DATA_DIR / split / cls).glob('*')))
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {split}/{cls}: {count} imágenes")
            if count == 0:
                all_ok = False
    
    print(f"\n  Total imágenes copiadas: {total_copied}")
    
    if all_ok:
        print("\n" + "=" * 60)
        print("  ✅ ¡DATASET LISTO PARA ENTRENAMIENTO!")
        print("=" * 60)
        print("  Ahora puedes ejecutar el Paso 7 (entrenar modelo)")
    else:
        print("\n  ⚠️ Algunas carpetas están vacías. Revisa los errores arriba.")
    
    return all_ok


# Ejecutar
if __name__ == "__main__":
    organize_kaggle_dataset_colab()
