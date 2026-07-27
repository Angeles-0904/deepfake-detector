"""
Punto de entrada principal para ejecutar el pipeline completo del sistema
de detección de deepfakes.

Uso:
    python run_pipeline.py --mode all          # Ejecutar todo
    python run_pipeline.py --mode data         # Solo preparar datos
    python run_pipeline.py --mode train        # Solo entrenar
    python run_pipeline.py --mode evaluate     # Solo evaluar
    python run_pipeline.py --mode robustness   # Solo pruebas de robustez
    python run_pipeline.py --mode app          # Iniciar interfaz web
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Detector - Pipeline completo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_pipeline.py --mode all           # Pipeline completo
  python run_pipeline.py --mode data          # Solo descargar y preparar datos
  python run_pipeline.py --mode train         # Solo entrenar modelo
  python run_pipeline.py --mode evaluate      # Solo evaluar modelo entrenado
  python run_pipeline.py --mode robustness    # Solo pruebas de robustez
  python run_pipeline.py --mode app           # Iniciar interfaz Streamlit
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "data", "train", "evaluate", "robustness", "app"],
        help="Modo de ejecución del pipeline",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Descargar dataset desde Kaggle (modo data)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint para evaluar (modo evaluate)",
    )

    args = parser.parse_args()

    if args.mode == "data":
        print("\n" + "=" * 60)
        print("  PIPELINE DE DATOS")
        print("=" * 60)
        from src.data_pipeline import run_pipeline
        run_pipeline(download=args.download)

    elif args.mode == "train":
        print("\n" + "=" * 60)
        print("  ENTRENAMIENTO DEL MODELO")
        print("=" * 60)
        from src.train import train
        history, test_metrics = train()

    elif args.mode == "evaluate":
        print("\n" + "=" * 60)
        print("  EVALUACIÓN DEL MODELO")
        print("=" * 60)
        from src.evaluate import run_evaluation
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
        results = run_evaluation(checkpoint_path=checkpoint_path)

    elif args.mode == "robustness":
        print("\n" + "=" * 60)
        print("  PRUEBAS DE ROBUSTEZ")
        print("=" * 60)
        from src.robustness import run_robustness_test
        results = run_robustness_test()

    elif args.mode == "all":
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETO")
        print("=" * 60)

        # 1. Datos
        print("\n--- Paso 1: Preparación de datos ---")
        from src.data_pipeline import run_pipeline
        run_pipeline(download=args.download)

        # 2. Entrenamiento
        print("\n--- Paso 2: Entrenamiento ---")
        from src.train import train
        history, test_metrics = train()

        # 3. Evaluación
        print("\n--- Paso 3: Evaluación ---")
        from src.evaluate import run_evaluation
        run_evaluation()

        # 4. Robustez
        print("\n--- Paso 4: Pruebas de robustez ---")
        from src.robustness import run_robustness_test
        run_robustness_test(
            baseline_acc=test_metrics["metrics"]["accuracy"]
            if test_metrics else None
        )

        print("\n✅ Pipeline completado exitosamente.")
        print(f"   Gráficos guardados en: outputs/plots/")
        print(f"   Modelo guardado en: outputs/checkpoints/")

    elif args.mode == "app":
        print("\n" + "=" * 60)
        print("  INICIANDO INTERFAZ WEB")
        print("=" * 60)
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).parent / "app" / "streamlit_app.py"),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
        ])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
