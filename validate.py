import math
from pathlib import Path

from tqdm.auto import tqdm

EXPECTED_SPLITS = {
    'train': {'total': 8222, 'positive': 2552, 'negative': 5670},
    'val': {'total': 1876, 'positive': 451, 'negative': 1425},
    'test': {'total': 3408, 'positive': 942, 'negative': 2466},
}


def validate_yolo_label(label_path: Path) -> int:
    lines = label_path.read_text(encoding='utf-8').splitlines()

    for line in lines:
        values = line.split()

        if len(values) != 5 or values[0] != '0':
            raise ValueError(f'Rótulo inválido: {label_path}')

        x, y, w, h = map(float, values[1:])

        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f'Valor não finito: {label_path}')

        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            raise ValueError(f'Coordenadas inválidas: {label_path}')

        if w <= 0 or h <= 0:
            raise ValueError(f'Dimensões inválidas: {label_path}')

        tolerance = 1e-6

        if (
            x - w / 2 < -tolerance
            or x + w / 2 > 1 + tolerance
            or y - h / 2 < -tolerance
            or y + h / 2 > 1 + tolerance
        ):
            raise ValueError(f'Caixa fora da imagem: {label_path}')

    return len(lines)


def validate_yolo_dataset(dataset_path: Path) -> None:
    for split, expected in EXPECTED_SPLITS.items():
        images_path = dataset_path / 'images' / split
        labels_path = dataset_path / 'labels' / split

        images = sorted(images_path.rglob('*.png'))

        positive = 0
        negative = 0
        total_boxes = 0

        for image_path in tqdm(images, desc=f'Validando {split}'):
            relative_path = image_path.relative_to(images_path)
            label_path = labels_path / relative_path.with_suffix('.txt')

            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            boxes = validate_yolo_label(label_path)

            positive += boxes > 0
            negative += boxes == 0
            total_boxes += boxes

        actual = {
            'total': len(images),
            'positive': positive,
            'negative': negative,
        }

        print(f'\n{split.upper()}')
        print(f'Imagens: {actual["total"]}/{expected["total"]}')
        print(f'Positivas: {positive}/{expected["positive"]}')
        print(f'Negativas: {negative}/{expected["negative"]}')
        print(f'Ácaros anotados: {total_boxes}')

        positive_ratio = positive / len(images) * 100 if images else 0

        expected_ratio = expected['positive'] / expected['total'] * 100

        print(
            f'Proporção positiva: {positive_ratio:.2f}% '
            f'(esperado: {expected_ratio:.2f}%)'
        )

        if actual != expected:
            raise ValueError(
                f'Distribuição incorreta no split {split}: {actual} != {expected}'
            )

        print('Validação concluída!')
