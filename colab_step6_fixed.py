"""
PASO 6 CORREGIDO - Ejecuta esta celda ANTES del Paso 7
Organiza el dataset de Kaggle correctamente en Colab
"""
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("  📥 ORGANIZANDO DATASET PARA ENTRENAMIENTO")
print("=" * 60)

# Rutas
KAGGLE_DIR = Path('/kaggle/input/140k-real-and-fake-faces')
PROJECT_DIR = Path('/content/drive/MyDrive/detectorIA')
DATA_DIR = PROJECT_DIR / 'data'

# Verificar dataset
if not KAGGLE_DIR.exists():
    print(f"❌ Dataset no encontrado en: {KAGGLE_DIR}")
    print("   Ejecuta el Paso 6 original primero")
else:
    print(f"✅ Dataset encontrado: {KAGGLE_DIR}")
    
    # Buscar imágenes
    images_dir = KAGGLE_DIR / 'real_vs_fake'
    if not images_dir.exists():
        print(f"❌ Carpeta real_vs_fake no encontrada")
    else:
        print(f"✅ Carpeta de imágenes: {images_dir}")
        
        # Crear carpetas destino
        for split in ['train', 'validation', 'test']:
            for cls in ['real', 'fake']:
                (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Mapeo de splits
        splits = {
            'train': (KAGGLE_DIR / 'train.csv', DATA_DIR / 'train'),
            'validation': (KAGGLE_DIR / 'valid.csv', DATA_DIR / 'validation'),
            'test': (KAGGLE_DIR / 'test.csv', DATA_DIR / 'test'),
        }
        
        total = 0
        
        for split_name, (csv_file, dest_base) in splits.items():
            print(f"\n📂 {split_name}...")
            
            if not csv_file.exists():
                print(f"  ❌ {csv_file.name} no encontrado")
                continue
            
            df = pd.read_csv(csv_file)
            
            # Buscar columnas
            path_col = next((c for c in df.columns if c.lower() == 'path'), None)
            label_col = next((c for c in df.columns if c.lower() == 'label'), None)
            
            if not path_col or not label_col:
                print(f"  ❌ Columnas no encontradas: {list(df.columns)}")
                continue
            
            real_dir = dest_base / 'real'
            fake_dir = dest_base / 'fake'
            
            real_count = 0
            fake_count = 0
            
            for _, row in tqdm(df.iterrows(), desc=f"  {split_name}", total=len(df)):
                img_path = str(row[path_col])
                label = int(row[label_col])
                
                src = images_dir / img_path
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
            
            total += real_count + fake_count
            print(f"  ✅ {real_count} reales, {fake_count} falsas")
        
        # Verificar
        print("\n" + "=" * 60)
        print("  📊 VERIFICACIÓN")
        print("=" * 60)
        
        all_ok = True
        for split in ['train', 'validation', 'test']:
            for cls in ['real', 'fake']:
                count = len(list((DATA_DIR / split / cls).glob('*')))
                status = "✅" if count > 0 else "❌"
                print(f"  {status} {split}/{cls}: {count}")
                if count == 0:
                    all_ok = False
        
        print(f"\n  Total: {total} imágenes")
        
        if all_ok:
            print("\n  ✅ ¡LISTO! Ejecuta ahora el Paso 7 (entrenar)")
        else:
            print("\n  ⚠️ Revisa los errores arriba")
