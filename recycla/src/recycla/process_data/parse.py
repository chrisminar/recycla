"""
This takes a directory of the format
dir
  dataset1
    class1
      class1img.jpg
      ...
    class2
    ...
    classn
  dataset2
  ...
  datasetn

it renames all .jpeg variations to .jpg
ignores anything that isn't .jpg or .png
        zero_count_classes = [class_name for class_name in all_classes
                             if val_class_counts[class_name] == 0 or test_class_counts[class_name] == 0]

        if zero_count_classes:
            print("\nFrames per class with zero counts in val or test:")
            for class_name in sorted(zero_count_classes):
                print(
                    f"{class_name}: train={train_class_counts[class_name]}, val={val_class_counts[class_name]}, test={test_class_counts[class_name]}"
                )nges names to be consistant
moves to new dir

newdirdir
  train
    metadata.csv
    class1
      class1img.jpg
      ...
    class2
    ...
    classn
  test
    ...
  val
"""

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path

import click
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Subset

from recycla import DATA_PATH, log
from recycla.process_data.io import import_file_hierarchy


def _get_available_datasets() -> list[Path]:
    """Retrieve a list of target directories based on the specified criteria.

    Returns:
        list[Path]: A list of Path objects representing the target directories.
    """
    return [o for o in (DATA_PATH / "raw_labeled_data").glob("*") if o.is_dir()]


def _purge_directory(targetpath: Path = DATA_PATH / "labeled_data") -> None:
    """Deletes the specified directory and all its contents.
    Args:
        targetpath (Path, optional): The path to the directory to be deleted. Defaults to DATA_PATH / "labeled_data".
    Returns:
        None
    Raises:
        None: This function suppresses FileNotFoundError if the directory does not exist.
    """
    try:
        shutil.rmtree(targetpath)
    except FileNotFoundError:
        pass


def _check_if_file_has_uid(file_path: Path) -> tuple[bool, str | None]:
    # uid ex 0h2E8QH6L8SPlGXZDqHK_8.jpg
    if "_" not in file_path.stem:
        return False, None
    parts = file_path.stem.split("_")
    if len(parts) != 2:
        return False, None
    if len(parts[0]) != 20:
        return False, None
    if not parts[0].isalnum():
        return False, None
    if not parts[1].isdigit():
        return False, None
    return True, parts[0]


class DataType(IntEnum):
    IMAGE = auto()
    VIDEO = auto()


@dataclass
class DataRecord:
    source: Path  # source file path
    dest: Path  # destination file path
    dataset: str
    dataType: DataType
    label: list[str]


def _find_date_range_path(target_dir: Path) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}$", target_dir.name))


def _generate_data_records(
    targets: list[Path], raw_map: dict[str, str]
) -> list[DataRecord]:
    """Generates a list of data records from the given target directories.

    Args:
        targets (list of Path): A list of Path objects representing the target directories.
        raw_map (dict): A dictionary mapping original labels to training labels.
    Returns:
        list of DataRecord: A list of DataRecord objects containing:
            - source_file (Path): The original file path.
            - destination_file_name (Path): The new file path with the class name and a unique number.
            - dataset_name (str): The name of the dataset directory.
            - label (str): The class label, which may contain multiple labels separated by commas.
    The function processes each target directory, iterating through all class folders and their images.
    It filters the images based on allowed suffixes (".png", ".jpg", ".jpeg"), renames them with a class-specific
    prefix and a unique number, and converts all JPEG variants to ".jpg". The resulting records contain the original
    file path, the new file path, and the dataset name.
    """
    jpegs = [".JPG", ".jpg", ".jpeg"]
    allowed_suffix_list = [".png"] + jpegs
    records = []
    numbers = defaultdict(lambda: 0)
    for dataset_dir in targets:  # ex: dataset1
        data_type = (
            DataType.VIDEO if _find_date_range_path(dataset_dir) else DataType.IMAGE
        )
        all_class_paths = [p for p in dataset_dir.glob("*") if p.is_dir()]
        for class_folder_with_images in all_class_paths:  # ex dataset1/class1
            files = class_folder_with_images.glob("*")
            cls = class_folder_with_images.name.lower()
            if cls not in raw_map:
                raise ValueError(f"Found invalid class name: {cls}.")
            mapped_cls = raw_map.get(cls)
            for source_file in files:  # ex dataset1/class1/image1.jpg
                # if file isn't an image, skip
                if source_file.suffix not in allowed_suffix_list:
                    continue

                # check if file has uid
                has_uid, _ = _check_if_file_has_uid(source_file)

                if has_uid:
                    # if it has a uid, use the original name
                    destination_file_name = source_file.name
                else:
                    # rename if no uid
                    destination_file_name = (
                        f"{mapped_cls}_{numbers[mapped_cls]}{source_file.suffix}"
                    )
                    numbers[mapped_cls] += 1
                destination_file_name = Path(mapped_cls) / destination_file_name
                if destination_file_name.suffix in jpegs:
                    destination_file_name = destination_file_name.with_suffix(".jpg")

                records.append(
                    DataRecord(
                        source=source_file,
                        dest=destination_file_name,
                        dataset=class_folder_with_images.parent.name,
                        dataType=data_type,
                        label=mapped_cls.split(", "),
                    )
                )
    return records


def _split_data_records_by_type(
    records: list[DataRecord],
) -> tuple[list[DataRecord], dict[str, list[DataRecord]]]:
    image_records = [r for r in records if r.dataType == DataType.IMAGE]
    video_records = [r for r in records if r.dataType == DataType.VIDEO]

    video_clusters = {}
    for record in video_records:
        _, video_uid = _check_if_file_has_uid(record.source)
        if video_uid is None:
            raise ValueError(f"Video file {record.source} does not have a valid UID.")
        if video_uid not in video_clusters:
            video_clusters[video_uid] = []
        video_clusters[video_uid].append(record)

    return image_records, video_clusters


def print_record_stats(
    records: list[DataRecord],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    # Print statistics about the split
    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)
    test_class_counts = defaultdict(int)

    if isinstance(records, list):
        for i in train_idx:
            # -1 will take the submaterial
            train_class_counts[str(", ").join(records[i].label)] += 1
        for i in val_idx:
            val_class_counts[str(", ").join(records[i].label)] += 1
        for i in test_idx:
            test_class_counts[str(", ").join(records[i].label)] += 1
    else:  # record cluster (videos)
        record_keys = list(records.keys())
        for i in train_idx:
            train_class_counts[str(", ").join(records[record_keys[i]][0].label)] += 1
        for i in val_idx:
            val_class_counts[str(", ").join(records[record_keys[i]][0].label)] += 1
        for i in test_idx:
            test_class_counts[str(", ").join(records[record_keys[i]][0].label)] += 1

    all_classes = (
        set(train_class_counts.keys())
        | set(val_class_counts.keys())
        | set(test_class_counts.keys())
    )

    # Only print classes with zero counts in val or test
    zero_val_classes = [
        class_name for class_name in all_classes if val_class_counts[class_name] == 0
    ]
    zero_test_classes = [
        class_name for class_name in all_classes if test_class_counts[class_name] == 0
    ]

    if zero_val_classes or zero_test_classes:
        print("\nClasses with zero counts:")
        for class_name in sorted(set(zero_val_classes + zero_test_classes)):
            print(
                f"{class_name}: train={train_class_counts[class_name]}, val={val_class_counts[class_name]}, test={test_class_counts[class_name]}"
            )

    # Check for classes with zero values in val and test sets
    missing_val_classes = []
    missing_test_classes = []

    for class_name in sorted(all_classes):
        if val_class_counts[class_name] == 0:
            missing_val_classes.append(class_name)
        if test_class_counts[class_name] == 0:
            missing_test_classes.append(class_name)

    return missing_val_classes, missing_test_classes


def _check_classes_not_empty(
    train_records: list[DataRecord],
    val_records: list[DataRecord],
    test_records: list[DataRecord],
) -> None:
    # Sometimes, if there are very few images there can be a missing folder
    test_classes = set([str(", ").join(r.label) for r in test_records])
    val_classes = set([str(", ").join(r.label) for r in val_records])
    train_classes = set([str(", ").join(r.label) for r in train_records])

    # assume train will have all classes
    missing_val_classes = train_classes - val_classes
    missing_test_classes = train_classes - test_classes
    if len(missing_val_classes) > 0 or len(missing_test_classes) > 0:
        raise ValueError(
            f"Missing classes in validation or test sets: {missing_val_classes}, {missing_test_classes}"
        )


def _split_data(
    records: list[DataRecord] | dict[str, list[DataRecord]],
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[Subset, Subset, Subset]:
    """Splits the given dataset into training, validation, and test subsets.
    Args:
        records: The dataset to be split, typically a list or array-like structure.
        split_fractions (tuple[float, float, float], optional): A tuple of (train_fraction, val_fraction, test_fraction).
                                                                Defaults to (0.8, 0.1, 0.1) (80% training, 10% validation, 10% test).
    Returns:
        tuple[Subset, Subset, Subset]: A tuple containing three subsets:
            - train_records: The training subset.
            - val_records: The validation subset.
            - test_records: The test subset.
    """
    train_fraction, val_fraction, test_fraction = split_fractions

    # Validate that fractions sum to 1.0
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_fraction + val_fraction + test_fraction}"
        )

    # First split: separate training from validation+test
    train_idx, val_test_idx = train_test_split(
        list(range(len(records))),
        test_size=val_fraction + test_fraction,
        random_state=42,
    )

    # Second split: separate validation from test
    # Calculate the relative size of test within the val+test portion
    test_relative_size = test_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=test_relative_size, random_state=42
    )

    missing_val_classes, missing_test_classes = print_record_stats(
        records, train_idx, val_idx, test_idx
    )

    if isinstance(records, dict):
        for phase, phase_index, missing_classes in zip(
            ["val", "test"],
            [val_idx, test_idx],
            [missing_val_classes, missing_test_classes],
        ):
            # for each missing class
            for missing_class in missing_classes:
                log.info(
                    f"Missing class {missing_class} in {phase} set. Attempting to move a video with this class from the train set."
                )
                # find the index of the missing class in the train set
                # phase_index is empty, so we will move a video to it
                # iterate over all videos
                # look for a video with the missing class to move to the empty phase
                for cluster_idx, video_uid in enumerate(records):
                    if cluster_idx not in train_idx:
                        continue  # skip if this video is in the train set
                    # get the class of this video
                    class_id = str(", ").join(records[video_uid][0].label)
                    # if the class matches a missing class
                    if class_id == missing_class:
                        index_to_move = train_idx.pop(train_idx.index(cluster_idx))
                        phase_index.append(index_to_move)  # move to the phase
                        log.info("success")
                        # only move one video per missing class
                        break
        print_record_stats(records, train_idx, val_idx, test_idx)

    if isinstance(records, dict):  # record cluster
        # rebuild list
        new_records = []
        new_indices = {"train": [], "val": [], "test": []}
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        overall_index = 0
        for cluster_idx, video_uid in enumerate(records):
            for phase, idx in splits.items():  # iterate over phase
                if cluster_idx in idx:  # check if this key is in the phase
                    # if it is, add all records to the new indices
                    for record in records[video_uid]:
                        new_indices[phase].append(overall_index)
                        new_records.append(record)
                        overall_index += 1
                    break  # if the phase was found, break out of the loop
        train_idx = new_indices["train"]
        val_idx = new_indices["val"]
        test_idx = new_indices["test"]
        records = new_records

    train_records = Subset(records, train_idx)
    val_records = Subset(records, val_idx)
    test_records = Subset(records, test_idx)
    return train_records, val_records, test_records


def _move_files_to_new_structure(
    records: list[DataRecord], phase: str, dest_dir: Path
) -> None:
    """Moves files to a new directory structure based on the provided records and phase.

    Args:
        records (list of DataRecord): A list of DataRecord objects containing the source file path,
                                  destination file path.
        phase (str): A string representing the phase, which will be used as a subdirectory in the destination path.
        dest_dir (Path): The root directory where the files will be moved to.
    Returns:
        None
    """
    # for source, dest, _, _clss in records:
    for record in records:
        dest = Path(dest_dir) / f"{phase}/{record.dest}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(record.source, dest)


def _make_metadata(records: list[DataRecord], phase: str, dest_dir: Path) -> None:
    """Generates a metadata CSV file from the given records.

    Args:
        records (list of DataRecord): A list of DataRecord objects containing the source file path,
                                       destination file path, and class labels.
                                  ignored, and the destination and class are
                                  written to the CSV file.
        phase (str): A string representing the phase, which is used as a
                     subdirectory name in the destination directory.
        dest_dir (Path or str): The root directory where the metadata CSV file
                                will be created. The phase subdirectory will be
                                appended to this path.
    Writes:
        A CSV file named 'metadata.csv' in the specified phase subdirectory
        within the destination directory. Each line in the CSV file contains
        the destination path and class from the records, separated by a comma.
    """
    with open(dest_dir / f"{phase}/metadata.csv", "w") as f:
        for record in records:
            dest = Path(dest_dir) / f"{phase}/{record.dest}"
            s = f"{dest}," + ", ".join(record.label) + f",{record.dataset}\n"
            f.write(s)


def _parse_data(split_fractions):
    log.info("Figuring out what to copy where...")
    raw_map = import_file_hierarchy()  # what labels to combine
    targets = _get_available_datasets()  # list of datasets
    records = _generate_data_records(targets, raw_map)  # all files

    log.info(f"Total data records generated: {len(records)}")

    image_records, video_records = _split_data_records_by_type(records)

    log.info(f"Data type breakdown:")
    log.info(f"  Image records: {len(image_records)}")
    log.info(
        f"  Video frame records: {sum(len(cluster) for cluster in video_records.values())}"
    )
    log.info(f"  Video clusters (unique videos): {len(video_records)}")

    _purge_directory()  # delete old labeled data

    log.info("Splitting data into train, val, and test sets...")
    train_image, val_image, test_image = _split_data(
        image_records, split_fractions=split_fractions
    )
    train_video, val_video, test_video = _split_data(
        video_records, split_fractions=split_fractions
    )

    # Combine image and video records
    train = ConcatDataset([train_image, train_video])
    val = ConcatDataset([val_image, val_video])
    test = ConcatDataset([test_image, test_video])

    _check_classes_not_empty(
        train, val, test
    )  # make sure all classes are present in all sets

    dest_path = DATA_PATH / "labeled_data"

    log.info("Copying data into new structure...")
    for recs, phase in zip([train, val, test], ["train", "val", "test"]):
        log.info(f"  Copying {phase} data...")
        _move_files_to_new_structure(recs, phase, dest_path)
        log.info(f"  making {phase} metadata.")
        _make_metadata(recs, phase, dest_path)
        log.info(f"  {phase} data copied.")

    log.info(f"Data parsing complete! New data structure created in {dest_path}.")


@click.command()
@click.option(
    "--train-fraction",
    default=0.8,
    help="Fraction of the dataset to include in the training set. Defaults to 0.8.",
)
@click.option(
    "--val-fraction",
    default=0.1,
    help="Fraction of the dataset to include in the validation set. Defaults to 0.1.",
)
@click.option(
    "--test-fraction",
    default=0.1,
    help="Fraction of the dataset to include in the test set. Defaults to 0.1.",
)
def parse_data(train_fraction, val_fraction, test_fraction):
    """
    Split labeled data into train, val, and test sets and copy to new structure.
    """
    split_fractions = (train_fraction, val_fraction, test_fraction)
    _parse_data(split_fractions)
