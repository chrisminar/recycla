import os
import re
from typing import Final

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from recycla import log

NORMALIZE_MEAN: Final = [0.485, 0.456, 0.406]
NORMALIZE_STD: Final = [0.229, 0.224, 0.225]
RESIZE_PIXELS: Final = 256
CROP_PIXELS: Final = 224


class SmartResize:
    """Custom resize that doesn't upscale small images like QVGA."""

    def __init__(self, target_size=224, min_resize_threshold=300):
        self.target_size = target_size
        self.min_resize_threshold = min_resize_threshold

    def __call__(self, img):
        # Handle tensor input
        if torch.is_tensor(img):
            # For tensors, get dimensions from shape: (batch, channels, height, width) or (channels, height, width)
            if img.dim() == 4:  # (n, 3, h, w)
                h, w = img.shape[2], img.shape[3]
            elif img.dim() == 3:  # (3, h, w)
                h, w = img.shape[1], img.shape[2]
            else:
                return img  # Unexpected shape, return as-is

            min_dim = min(w, h)

            if min_dim <= self.target_size:
                # Small images: resize directly to target
                return transforms.Resize((self.target_size, self.target_size))(img)
            elif min_dim < self.min_resize_threshold:
                # Medium images: resize shorter edge to target_size, then crop
                return transforms.Compose(
                    [
                        transforms.Resize(self.target_size),
                        transforms.CenterCrop(self.target_size),
                    ]
                )(img)
            else:
                # Large images: use standard approach (resize to larger size, then crop)
                return transforms.Compose(
                    [
                        transforms.Resize(
                            int(self.target_size * 1.14)
                        ),  # ~256 for 224 target
                        transforms.CenterCrop(self.target_size),
                    ]
                )(img)

        # Handle PIL image input
        w, h = img.size
        min_dim = min(w, h)

        if min_dim <= self.target_size:
            # Small images (like QVGA): resize directly to target without upscaling beyond necessity
            return transforms.Resize((self.target_size, self.target_size))(img)
        elif min_dim < self.min_resize_threshold:
            # Medium images: resize shorter edge to target_size, then crop
            return transforms.Compose(
                [
                    transforms.Resize(self.target_size),
                    transforms.CenterCrop(self.target_size),
                ]
            )(img)
        else:
            # Large images: use standard approach (resize to larger size, then crop)
            return transforms.Compose(
                [
                    transforms.Resize(
                        int(self.target_size * 1.14)
                    ),  # ~256 for 224 target
                    transforms.CenterCrop(self.target_size),
                ]
            )(img)


smart_resize = SmartResize(CROP_PIXELS)


class TrainingAugmentations:
    """Custom training augmentations for recycling classification."""

    def __init__(
        self,
        crop_size=224,
        crop_scale=(0.8, 1.0),
        horizontal_flip_p=0.5,
        # Color augmentation parameters
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        grayscale_p=0.1,
        # Geometric augmentation parameters
        rotation_degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        # Quality augmentation parameters
        blur_kernel_size=3,
        blur_sigma=(0.1, 2.0),
        # Erasing parameters
        erasing_p=0.1,
        erasing_scale=(0.02, 0.33),
        erasing_ratio=(0.3, 3.3),
        # Normalization
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        # Flags to enable/disable transforms
        enable_crop=True,
        enable_horizontal_flip=True,
        enable_color_jitter=False,
        enable_grayscale=False,
        enable_rotation=False,
        enable_affine=False,
        enable_blur=False,
        enable_erasing=False,
    ):
        transform_list = []

        # Geometric transformations
        if enable_crop:
            transform_list.append(
                transforms.RandomResizedCrop(crop_size, scale=crop_scale)
            )
        else:
            transform_list.append(transforms.Resize((crop_size, crop_size)))

        if enable_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_p))

        # Color augmentations
        if enable_color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

        if enable_grayscale:
            transform_list.append(transforms.RandomGrayscale(p=grayscale_p))

        # Geometric augmentations
        if enable_rotation:
            transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

        if enable_affine:
            transform_list.append(
                transforms.RandomAffine(degrees=0, translate=translate, scale=scale)
            )

        # Quality augmentations
        if enable_blur:
            transform_list.append(
                transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
            )

        # Convert to tensor (always required)
        transform_list.append(transforms.ToTensor())

        # Post-tensor augmentations
        if enable_erasing:
            transform_list.append(
                transforms.RandomErasing(
                    p=erasing_p, scale=erasing_scale, ratio=erasing_ratio
                )
            )

        # Normalization (always required)
        transform_list.append(transforms.Normalize(normalize_mean, normalize_std))

        self.transforms = transforms.Compose(transform_list)

    def __call__(self, img):
        return self.transforms(img)


# Create augmentation instances
training_augmentations = TrainingAugmentations()

DATA_TRANSFORMS: Final = {
    "test": transforms.Compose(
        [
            smart_resize,
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    ),
    "train": training_augmentations,
    "val": transforms.Compose(
        [
            smart_resize,
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    ),
}


class CustomImageFolder(datasets.ImageFolder):
    def __init__(
        self, root, transform=None, target_transform=None, specified_classes=None
    ):
        self.specified_classes = specified_classes
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            allow_empty=True,
        )

        classes = self.classes
        primary_labels = [label.split(", ")[0] for label in classes]
        secondary_labels = classes
        primary_label_encoder = LabelEncoder()
        secondary_label_encoder = LabelEncoder()
        primary_label_encoder.fit(primary_labels)
        secondary_label_encoder.fit(secondary_labels)
        primary_class_names = primary_label_encoder.classes_
        secondary_class_names = secondary_label_encoder.classes_
        primary_to_idx = {cls_name: i for i, cls_name in enumerate(primary_class_names)}
        idx_to_labels = []
        for i, cls_name in enumerate(secondary_class_names):
            idx_to_labels.append(
                (
                    primary_to_idx[primary_labels[i]],
                    i,
                )
            )
        self.idx_to_labels = idx_to_labels
        self.primary_class_names = primary_class_names
        self.secondary_class_names = secondary_class_names

    def find_classes(self, directory):
        if self.specified_classes is not None:
            classes = self.specified_classes
        else:
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = self.samples[index][1]

        # Split the target into primary and secondary labels
        primary_label, secondary_label = self.idx_to_labels[target]

        return sample, (primary_label, secondary_label)


class RecyclaDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        phase: str,
        classnames: list[str] | None = None,
        transform_kwargs: dict | None = None,
    ):
        self.data_dir = data_dir
        self.phase = phase
        self.metadata = {"file_path": [], "dataset": []}
        metadata_file = os.path.join(data_dir, phase, "metadata.csv")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                for line in f:
                    d = re.split(r",(?=\S)", line.strip())
                    self.metadata["file_path"].append(d[0])
                    self.metadata["dataset"].append(d[2])
        self.dataset = CustomImageFolder(
            os.path.join(data_dir, phase),
            (
                DATA_TRANSFORMS[phase]
                if phase != "train"
                else (
                    TrainingAugmentations(**transform_kwargs)
                    if transform_kwargs
                    else TrainingAugmentations()
                )
            ),
            specified_classes=classnames,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        dataset = self.metadata["dataset"][idx]
        return data, label, dataset


def data_load(
    data_dir: str,
    phases: list[str],
    classnames: list[str] | None = None,
    transform_kwargs: dict | None = None,
    n_params: int = 2e6,
) -> tuple[
    dict[str, torch.utils.data.DataLoader],
    dict[str, int],
    list[str],
    list[str],
]:
    """
    Loads data from the specified directory and returns dataloaders, dataset sizes, and class names.

    Args:
        data_dir (str): The directory where the data is stored.
        phases (list[str]): list of phases (e.g., ['train', 'val', 'test']) to load data for.
        classnames


    Returns:
        tuple[dict[str, torch.utils.data.DataLoader], dict[str, int], list[str]]:
            - A dictionary where keys are phases and values are DataLoader objects for each phase.
            - A dictionary where keys are phases and values are the sizes of the datasets for each phase.
            - A list of primary class names.
            - A list of secondary class names.
    """

    image_datasets = {
        p: RecyclaDataset(os.path.join(data_dir), p, classnames, transform_kwargs)
        for p in phases
    }

    # Intelligent worker selection based on GPU and model complexity
    def determine_num_workers(n_params: int) -> tuple[int, int]:
        """
        Determine optimal number of workers based on GPU capabilities and model parameters.

        Args:
            n_params: Number of model parameters

        Returns:
            tuple[int, int]: (train_workers, other_workers)
        """
        # Check if CUDA is available and get GPU info
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0

        # Get GPU memory info if available
        total_gpu_memory = 0
        if gpu_available:
            for i in range(gpu_count):
                total_gpu_memory += torch.cuda.get_device_properties(i).total_memory

        # Convert GPU memory to GB
        total_gpu_memory_gb = total_gpu_memory / (1024**3)  # ~ 16GB

        # Decision logic for number of workers
        if not gpu_available:
            # No GPU: use minimal workers
            train_workers = 1
            other_workers = 1
        else:
            # large models don't need async loading to max out GPU utilization
            if n_params >= 10e6:
                train_workers = 1
            else:
                train_workers = int(20 * total_gpu_memory_gb * 1e6 // n_params)
            train_workers = max(
                min(train_workers, 14), 1
            )  # Cap at 14 workers and min 1
            other_workers = 2

        # Apply constraints: training workers should not exceed 14
        train_workers = min(train_workers, 14)

        # If training workers is 1, other workers should also be 1
        if train_workers == 1:
            other_workers = 1

        return train_workers, other_workers

    # Determine optimal worker counts
    train_workers, other_workers = determine_num_workers(n_params)
    # train_workers, other_workers = 0, 0
    log.warning(
        f"Using {train_workers} workers for training and {other_workers} for other phases based on model size: {n_params / 1e6:.2f}M parameters"
    )

    dataloaders = {
        p: torch.utils.data.DataLoader(
            image_datasets[p],
            batch_size=64,
            shuffle=True,
            num_workers=train_workers if p == "train" else other_workers,
            pin_memory=train_workers > 0,  # only pin memory if multiple workers
            persistent_workers=train_workers
            > 0,  # only use persistent workers if multiple workers
        )
        for p in phases
    }
    dataset_sizes = {p: len(image_datasets[p]) for p in phases}
    primary_class_names = image_datasets[phases[0]].dataset.primary_class_names
    secondary_class_names = image_datasets[phases[0]].dataset.secondary_class_names

    return (
        dataloaders,
        dataset_sizes,
        primary_class_names,
        secondary_class_names,
    )
