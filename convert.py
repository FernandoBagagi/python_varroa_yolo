import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm.auto import tqdm

from download import SPLIT_NAMES


def _read_original_labels(label_path: Path) -> list[tuple[float, float, float, float]]:

    lines = label_path.read_text(encoding='utf-8').splitlines()

    if not lines:
        raise ValueError(f'Anotação vazia: {label_path}')

    image_class = lines[0].strip()

    if image_class not in {'0', '1', '3'}:
        raise ValueError(f'Classe inválida: {label_path}')

    image_class = 1 if int(image_class) != 0 else 0

    boxes = []

    for line in lines[1:]:
        if not line.strip():
            continue

        values = tuple(float(value) for value in line.split())

        if len(values) != 4 != 0:
            raise ValueError(f'Quantidade inválida de coordenadas: {label_path}')

        boxes.append(values)

    if image_class == 0 and boxes:
        raise ValueError(f'Imagem negativa com caixas: {label_path}')

    if image_class == 1 and not boxes:
        raise ValueError(f'Imagem positiva sem caixas: {label_path}')

    return boxes


def _xyxy_to_yolo(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = box

    if not (0 <= x_min < x_max <= width):
        raise ValueError(f'Coordenadas X inválidas: {box}')

    if not (0 <= y_min < y_max <= height):
        raise ValueError(f'Coordenadas Y inválidas: {box}')

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    return x_center, y_center, box_width, box_height


def _convert_label(
    label_path: Path, destination_path: Path, width: int, height: int
) -> None:

    boxes = _read_original_labels(label_path)

    yolo_lines: list[str] = []

    for box in boxes:
        coordinates = _xyxy_to_yolo(box, width, height)

        values = ' '.join(f'{value:.5f}' for value in coordinates)

        yolo_lines.append(f'0 {values}')

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    content = '\n'.join(yolo_lines)

    if content:
        content += '\n'

    destination_path.write_text(content, encoding='utf-8')


def _convert_split(dataset_path: Path, destination_path: Path, split_name: str) -> None:

    images_source_dir_path = dataset_path / split_name / 'videos'
    labels_source_dir_path = dataset_path / split_name / 'labels'

    images_destination_path = destination_path / 'images' / split_name
    labels_destination_path = destination_path / 'labels' / split_name

    invalid_image = (
        images_source_dir_path
        / '2017-08-30_15-42-59'
        / '2017-08-30_15-42-59.mp4-bee_id_11104-105630-1.png'
    )

    image_paths = sorted(images_source_dir_path.rglob('*.png'))

    if invalid_image in image_paths:
        image_paths.remove(invalid_image)

    if not image_paths:
        raise FileNotFoundError(f'Nenhuma imagem encontrada: {images_source_dir_path}')

    for image_path in tqdm(image_paths, desc=f'Convertendo {split_name}'):
        relative_path = image_path.relative_to(images_source_dir_path)

        label_path = labels_source_dir_path / relative_path.with_suffix('.txt')

        if not label_path.is_file():
            raise FileNotFoundError(f'Anotação não encontrada: {label_path}')

        output_image_path = images_destination_path / relative_path
        output_label_path = labels_destination_path / relative_path.with_suffix('.txt')

        with Image.open(image_path) as image:
            width, height = image.size

        _convert_label(label_path, output_label_path, width, height)

        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, output_image_path)


def prepare_yolo_dataset(dataset_path: Path, destination_path: Path) -> Path:

    for split_name in SPLIT_NAMES:
        _convert_split(dataset_path, destination_path, split_name)

    config = {
        'path': str(destination_path.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'varroa'},
    }

    yaml_path = destination_path / 'data.yaml'

    with yaml_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    print('Conversão concluída!')
    print(f'Dataset YOLO: {destination_path}')

    return yaml_path


def find_invalid_boxes(dataset_path: Path) -> list[dict]:
    invalid_boxes = []

    for split in ('train', 'val', 'test'):
        images_path = dataset_path / split / 'videos'
        labels_path = dataset_path / split / 'labels'

        images = sorted(images_path.rglob('*.png'))

        for image_path in tqdm(images, desc=f'Verificando {split}'):
            relative = image_path.relative_to(images_path)
            label_path = labels_path / relative.with_suffix('.txt')

            boxes = _read_original_labels(label_path)

            with Image.open(image_path) as image:
                width, height = image.size

            for box in boxes:
                x_min, y_min, x_max, y_max = box

                valid = 0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height

                if not valid:
                    invalid_boxes.append(
                        {
                            'split': split,
                            'image': str(image_path),
                            'label': str(label_path),
                            'box': box,
                        }
                    )

    return invalid_boxes
