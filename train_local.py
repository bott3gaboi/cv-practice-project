from ultralytics import YOLO
from pathlib import Path
from multiprocessing import freeze_support


def main():
    BASE_DIR = Path(__file__).resolve().parent

    DATA_PATH = BASE_DIR / "data" / "data.yaml"
    PROJECT_DIR = BASE_DIR / "results" / "local_training"

    models = [
        ("yolo11n.pt", "yolo11n_emergency_50ep", 16),
        ("yolo11s.pt", "yolo11s_emergency_50ep", 16),
        ("yolov8l.pt", "yolov8l_emergency_50ep_local", 8),
    ]

    print("DATA_PATH:", DATA_PATH)
    print("PROJECT_DIR:", PROJECT_DIR)

    for model_name, run_name, batch_size in models:
        print(f"\n===== Training {run_name} =====")

        model = YOLO(model_name)

        model.train(
            data=str(DATA_PATH),
            epochs=50,
            imgsz=640,
            batch=batch_size,
            cache=True,
            project=str(PROJECT_DIR),
            name=run_name,
            device=0,
            workers=0
        )

        print(f"===== Finished {run_name} =====")


if __name__ == "__main__":
    freeze_support()
    main()