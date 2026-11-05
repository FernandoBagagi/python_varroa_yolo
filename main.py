from pathlib import Path

from ultralytics import YOLO

from convert import prepare_yolo_dataset
from download import ensure_dataset
from validate import validate_yolo_dataset

dataset_path = ensure_dataset(Path('dataset'))

yolo_dataset_path = Path('dataset_yolo')

yaml_path = prepare_yolo_dataset(dataset_path, yolo_dataset_path)

validate_yolo_dataset(yolo_dataset_path)

model = YOLO('yolo11n.pt')

results = model.train(
    data='dataset_yolo/data.yaml',
    epochs=15,
    imgsz=320,
    batch=4,
    device='cpu',
    workers=2,
    cache=False,
    amp=False,
    project='runs/varroa',
    name='test_01',
)
