"""Prepare CUB, ImageNet-A, or ImageNet-R as train/<class>/ and test/<class>/.

Datasets with a published train/test split keep that split. When a source has
only one split (for example ImageNet-R), this utility makes a reproducible,
stratified split for experiments and smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import random
import shutil
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlparse
import zipfile
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = {
    "cub": 200, "imageneta": 200, "imagenetr": 200,
    "vtab": 50, "omnibench": 300, "omnibenchmark": 300,
    "cifar100": 100, "cifar224": 100,
}
DATASET_DIRS = {
    "cub": "cub", "imageneta": "imagenet-a", "imagenetr": "imagenet-r",
    "vtab": "vtab", "omnibench": "omnibenchmark", "omnibenchmark": "omnibenchmark",
    "cifar100": "cifar224", "cifar224": "cifar224",
}
RAW_NAMES = {
    "cub": "cub", "imageneta": "imagenet_a", "imagenetr": "imagenet_r",
    "vtab": "vtab", "omnibench": "omnibenchmark", "omnibenchmark": "omnibenchmark",
    "cifar100": "cifar100", "cifar224": "cifar100",
}


def _split_indices(labels: list[int], test_fraction: float, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)
    train: set[int] = set()
    test: set[int] = set()
    for indices in by_label.values():
        if len(indices) < 2:
            raise ValueError("Every class needs at least two images to create train and test splits")
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, round(len(indices) * test_fraction)))
        test.update(indices[:n_test])
        train.update(indices[n_test:])
    return train, test


def _class_directories(root: Path) -> dict[str, list[Path]]:
    """Find a class-folder tree, including one wrapper directory from tar files."""
    best: dict[str, list[Path]] = {}
    for candidate in (root, *sorted(p for p in root.iterdir() if p.is_dir())):
        classes = {
            class_dir.name: sorted(
                p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for class_dir in candidate.iterdir()
            if class_dir.is_dir()
        }
        classes = {name: paths for name, paths in classes.items() if paths}
        if len(classes) > len(best):
            best = classes
    return best


def _copy_imagefolder(source: Path, output: Path) -> bool:
    """Copy an already split ImageFolder without silently re-splitting it."""
    train, test = source / "train", source / "test"
    if not (train.is_dir() and test.is_dir()):
        return False
    train_classes = {p.name for p in train.iterdir() if p.is_dir()}
    test_classes = {p.name for p in test.iterdir() if p.is_dir()}
    if not train_classes or train_classes != test_classes:
        raise ValueError("Source train/test directories must contain the same non-empty class set")
    for split in ("train", "test"):
        for image in (source / split).rglob("*"):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                relative = image.relative_to(source / split)
                destination = output / split / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, destination)
    return True


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract zip/tar archives while rejecting path traversal entries."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) if zipfile.is_zipfile(archive) else tarfile.open(archive) as handle:
        members = handle.infolist() if isinstance(handle, zipfile.ZipFile) else handle.getmembers()
        for member in members:
            name = member.filename
            if isinstance(handle, zipfile.ZipFile):
                is_directory = name.endswith("/")
                is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
            else:
                is_directory = member.isdir()
                is_symlink = member.issym() or member.islnk()
            unsupported_tar_member = not isinstance(handle, zipfile.ZipFile) and not (member.isdir() or member.isreg())
            if is_symlink or unsupported_tar_member:
                raise ValueError(f"Archive contains unsupported link or special file: {name}")
            target = (destination / name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Archive contains unsafe path: {name}")
        handle.extractall(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source(source: Path | None, url: str | None, cache: Path | None, checksum: str | None):
    """Resolve a directory or download/extract one archive into a temporary tree."""
    if source is not None and source.exists() and source.is_dir():
        return source, None
    if source is not None and not source.is_file() and url is None:
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if source is not None and source.is_file():
        if checksum and _sha256(source).lower() != checksum.lower():
            raise ValueError(f"SHA256 mismatch for {source}")
        temporary_dir = Path(tempfile.mkdtemp(prefix="prepared-dataset-"))
        _safe_extract(source, temporary_dir)
        return temporary_dir, temporary_dir
    if url is None:
        raise FileNotFoundError("Provide an existing --source directory/archive or --url")
    cache = cache or Path(".dataset-cache")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / Path(urlparse(url).path).name
    if not archive.name or archive.name == ".":
        archive = cache / "dataset.download"
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(archive)
    if checksum and _sha256(archive).lower() != checksum.lower():
        raise ValueError(f"SHA256 mismatch for {archive}")
    temporary_dir = Path(tempfile.mkdtemp(prefix="prepared-dataset-"))
    _safe_extract(archive, temporary_dir)
    return temporary_dir, temporary_dir


def _prepare_cub_official(source: Path, output: Path) -> bool:
    """Use CUB_200_2011's published train_test_split.txt if present."""
    roots = [source, *sorted(p for p in source.rglob("CUB_200_2011") if p.is_dir())]
    for root in roots:
        image_root = root / "images"
        split_file = root / "train_test_split.txt"
        image_file = root / "images.txt"
        if not (image_root.is_dir() and split_file.is_file() and image_file.is_file()):
            continue
        image_relpaths: dict[int, str] = {}
        for line in image_file.read_text(encoding="utf-8").splitlines():
            image_id, relpath = line.split(maxsplit=1)
            image_relpaths[int(image_id)] = relpath
        split_ids: dict[int, int] = {}
        for line in split_file.read_text(encoding="utf-8").splitlines():
            image_id, is_train = line.split()
            split_ids[int(image_id)] = int(is_train)
        if set(image_relpaths) != set(split_ids):
            raise ValueError("CUB images.txt and train_test_split.txt contain different image IDs")
        for image_id, relpath in image_relpaths.items():
            split = "train" if split_ids[image_id] else "test"
            source_image = image_root / relpath
            if not source_image.is_file():
                raise FileNotFoundError(f"CUB image listed in metadata is missing: {source_image}")
            destination = output / split / Path(relpath).parent / Path(relpath).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
        return True
    return False


def _prepare_parquet(source: Path, output: Path, test_fraction: float, seed: int) -> bool:
    parquet_files = sorted(source.rglob("*.parquet"))
    if not parquet_files:
        return False
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Parquet sources require `pip install datasets`.") from exc

    split_files: dict[str, list[Path]] = defaultdict(list)
    for path in parquet_files:
        parent_names = {part.lower() for part in path.relative_to(source).parts[:-1]}
        split_files["test" if "test" in parent_names else "train"].append(path)

    datasets_by_split = {
        split: load_dataset("parquet", data_files=[str(p) for p in files], split="train")
        for split, files in split_files.items()
    }
    first_dataset = next(iter(datasets_by_split.values()))
    if "image" not in first_dataset.features:
        raise ValueError(f"Expected an image column; found {list(first_dataset.features)}")
    label_column = next(
        (column for column in ("label", "class_name", "class", "wnid") if column in first_dataset.features),
        None,
    )
    if label_column is None:
        raise ValueError(
            f"Expected a label, class_name, class, or wnid column; found {list(first_dataset.features)}"
        )

    all_labels = [label for dataset in datasets_by_split.values() for label in dataset[label_column]]
    feature_names = getattr(first_dataset.features[label_column], "names", None)
    metadata_names = _class_names_from_metadata(source) if label_column == "label" else None
    if all(isinstance(label, str) for label in all_labels):
        # ImageNet parquet exports commonly store the class directory/wnid as text.
        # Use those values directly, in stable order, instead of treating them as IDs.
        names = sorted(set(all_labels))
        label_to_name = {name: name for name in names}
    else:
        try:
            numeric_labels = [int(label) for label in all_labels]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {label_column!r} must contain all strings or numeric class IDs") from exc
        names = feature_names or metadata_names
        if not names:
            # Preserve numeric IDs with stable generated class folder names.
            names = [f"class_{index:03d}" for index in range(max(numeric_labels, default=-1) + 1)]
        if numeric_labels and (min(numeric_labels) < 0 or max(numeric_labels) >= len(names)):
            raise ValueError(
                f"Label IDs range from {min(numeric_labels)} to {max(numeric_labels)}, "
                f"but only {len(names)} class names were found"
            )
        label_to_name = {index: name for index, name in enumerate(names)}

    if set(datasets_by_split) == {"train", "test"}:
        selected = [(split, dataset, range(len(dataset))) for split, dataset in datasets_by_split.items()]
    else:
        dataset = first_dataset
        labels = dataset[label_column]
        split_labels = [names.index(label) if isinstance(label, str) else int(label) for label in labels]
        train_ids, test_ids = _split_indices(split_labels, test_fraction, seed)
        selected = [("train", dataset, sorted(train_ids)), ("test", dataset, sorted(test_ids))]
    for split, dataset, indices in selected:
        for index in indices:
            label = dataset[index][label_column]
            image = _decode_image(dataset[index]["image"], index)
            destination = output / split / label_to_name[label] / f"{index:06d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(destination, quality=95)
    return True


def _decode_image(value: object, index: int):
    """Decode Hugging Face Image values whether returned as PIL or Arrow records."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        image_path = value.get("path")
        if raw_bytes:
            return Image.open(BytesIO(raw_bytes))
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                return Image.open(path)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(value)))
    elif isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(
        f"Dataset row {index} image has unsupported value type {type(value).__name__}; "
        "expected a PIL image or a record with image bytes/path"
    )


def _class_names_from_metadata(source: Path) -> list[str] | None:
    """Read Hugging Face ClassLabel names retained in dataset_infos.json."""
    metadata_path = source / "dataset_infos.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def find_names(value: object) -> list[str] | None:
        if isinstance(value, dict):
            label = value.get("label")
            if isinstance(label, dict) and isinstance(label.get("names"), list):
                names = label["names"]
                if all(isinstance(name, str) for name in names):
                    return names
            for child in value.values():
                found = find_names(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_names(child)
                if found:
                    return found
        return None

    return find_names(metadata)


def _prepare_class_folders(source: Path, output: Path, test_fraction: float, seed: int) -> bool:
    classes = _class_directories(source)
    if not classes:
        return False
    names = sorted(classes)
    paths = [path for name in names for path in classes[name]]
    labels = [label for label, name in enumerate(names) for _ in classes[name]]
    train_ids, test_ids = _split_indices(labels, test_fraction, seed)
    for split, indices in (("train", train_ids), ("test", test_ids)):
        for index in sorted(indices):
            name = names[labels[index]]
            source_image = paths[index]
            destination = output / split / name / source_image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="imageneta", choices=sorted(EXPECTED_CLASSES))
    parser.add_argument("--source", help="Downloaded archive, extracted folders, or HF repository")
    parser.add_argument("--url", help="URL of an archive to download when --source is absent")
    parser.add_argument("--cache", help="Archive cache directory used with --url")
    parser.add_argument("--sha256", help="Expected SHA256 for --source/--url archive")
    parser.add_argument("--output", help="Prepared dataset directory (default: data/datasets/<dataset>)")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test_fraction must be between 0 and 1")

    source_arg = Path(args.source).expanduser().resolve() if args.source else None
    if source_arg is None:
        source_candidates = [
            Path("/kaggle/working/hf-datasets/raw") / RAW_NAMES[args.dataset],
            Path("data/raw") / RAW_NAMES[args.dataset],
            Path("data/raw") / args.dataset,
        ]
        source_arg = next((candidate.resolve() for candidate in source_candidates if candidate.exists()), None)
        if source_arg is None and args.url is None:
            searched = ", ".join(str(path) for path in source_candidates)
            raise FileNotFoundError(
                f"Could not infer dataset source. Searched: {searched}. "
                "Pass --source or download the Hugging Face snapshot first."
            )
    output = Path(args.output or (Path("data/datasets") / DATASET_DIRS[args.dataset])).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    source, temporary_source = _resolve_source(
        source_arg, args.url, Path(args.cache).expanduser().resolve() if args.cache else None, args.sha256
    )
    try:
        prepared = _copy_imagefolder(source, output)
        if not prepared and args.dataset == "cub":
            prepared = _prepare_cub_official(source, output)
        if not prepared:
            prepared = _prepare_parquet(source, output, args.test_fraction, args.seed)
        if not prepared:
            prepared = _prepare_class_folders(source, output, args.test_fraction, args.seed)
        if not prepared:
            raise FileNotFoundError(
                f"Could not find CUB metadata, Parquet files, or class folders under {source}"
            )
    finally:
        if temporary_source is not None:
            shutil.rmtree(temporary_source, ignore_errors=True)

    train_classes = {p.name for p in (output / "train").iterdir() if p.is_dir()}
    test_classes = {p.name for p in (output / "test").iterdir() if p.is_dir()}
    expected = EXPECTED_CLASSES[args.dataset]
    if len(train_classes) != expected or train_classes != test_classes:
        raise ValueError(
            f"Prepared {args.dataset} has {len(train_classes)} train and {len(test_classes)} test classes; "
            f"expected matching sets of {expected}. Check source dataset and split metadata."
        )
    print(f"Prepared {args.dataset}: {len(train_classes)} classes at {output}")


if __name__ == "__main__":
    main()
