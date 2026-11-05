import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

DATASET_URL = 'https://zenodo.org/records/4085044/files/'

SPLIT_NAMES = ('train', 'val', 'test')


def _is_split_path_valid(dataset_path: Path, split_name: str) -> bool:

    split_path = dataset_path / split_name
    images_path = split_path / 'videos'
    annotations_path = split_path / 'gt_one.csv'

    return (
        images_path.is_dir()
        and annotations_path.is_file()
        and any(images_path.rglob('*.png'))
    )


def _is_dataset_valid(dataset_path: Path) -> bool:

    if not dataset_path.is_dir():
        return False

    if not (dataset_path / 'gt.csv').is_file():
        return False

    return all(_is_split_path_valid(dataset_path, split) for split in SPLIT_NAMES)


def _is_zip_valid(zip_path: Path) -> bool:
    if not zip_path.is_file():
        return False

    try:
        with zipfile.ZipFile(zip_path) as z:
            return z.testzip() is None
    except OSError, zipfile.BadZipFile:
        return False


def _extract_zip(zip_path: Path, extract_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()

        with tqdm(
            total=len(members),
            desc=f'Extraindo {zip_path.name}',
            unit='arquivo',
        ) as progress:
            for member in members:
                z.extract(member, extract_path)
                progress.update(1)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    temp = target.with_suffix(target.suffix + '.part')

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get('Content-Length', 0))

            with (
                temp.open('wb') as file,
                tqdm(
                    total=total or None,
                    desc=target.name,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress,
            ):
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    file.write(chunk)
                    progress.update(len(chunk))

        if total and temp.stat().st_size != total:
            raise OSError(f'Download incompleto: {target.name}')

        temp.replace(target)

    finally:
        temp.unlink(missing_ok=True)


def ensure_dataset(dataset_path: Path = Path('dataset')) -> Path:

    dataset_path = Path(dataset_path)

    if _is_dataset_valid(dataset_path):
        print('Dataset encontrado. Pulando download.')
        return dataset_path

    dataset_path.mkdir(parents=True, exist_ok=True)

    gt_path = dataset_path / 'gt.csv'

    if not gt_path.is_file():
        print('Baixando gt.csv...')
        _download(DATASET_URL + 'gt.csv?download=1', gt_path)

    for split in SPLIT_NAMES:
        if _is_split_path_valid(dataset_path, split):
            print(f'{split}: arquivos encontrados.')
            continue

        zip_path = dataset_path / f'{split}.zip'

        if not _is_zip_valid(zip_path):
            zip_path.unlink(missing_ok=True)

            print(f'Baixando {split}.zip...')
            _download(DATASET_URL + f'{split}.zip?download=1', zip_path)

        if not _is_zip_valid(zip_path):
            raise RuntimeError(f'Arquivo ZIP inválido: {zip_path}')

        print(f'Extraindo {split}.zip...')
        _extract_zip(zip_path, dataset_path)

        if not _is_split_path_valid(dataset_path, split):
            raise RuntimeError(f'Falha na validação do conjunto: {split}')

        zip_path.unlink()
        print(f'{split}: extração concluída.')

    if not _is_dataset_valid(dataset_path):
        raise RuntimeError('Dataset incompleto.')

    print('Dataset preparado com sucesso!')
    return dataset_path
