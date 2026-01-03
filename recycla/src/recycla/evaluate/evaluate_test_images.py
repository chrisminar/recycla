from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from recycla import device, log
from recycla.classify.classification_utils import PredictionPreparation
from recycla.evaluate.statistics import calculate_accuracy

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _count_class_correct_test_images(
    model: nn.Module,
    dataloader: data.DataLoader,
    dataset_size: int,
    primary_classnames: list[str],
    secondary_classnames: list[str],
) -> tuple[
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluates the accuracy of a classification model for each class and overall.
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloaders (dict): A dictionary containing the dataloaders for each phase (e.g., 'train', 'val', 'test').
        dataset_sizes (dict): A dictionary containing the sizes of the datasets for each phase.
        classnames (list): A list of class names corresponding to the classes in the dataset.
    Returns:
        tuple[list[float], float, np.ndarray]: A tuple containing the accuracy for each class, the overall accuracy, and the sample counts for each class.
    """
    model.eval()

    primary_class_counts = defaultdict(lambda: 0)
    secondary_class_counts = defaultdict(lambda: 0)
    primary_class_correct = defaultdict(
        lambda: torch.zeros(len(primary_classnames), device=torch.device("cpu"))
    )
    secondary_class_correct = defaultdict(
        lambda: torch.zeros(len(secondary_classnames), device=torch.device("cpu"))
    )

    all_primary_labels = []
    all_primary_preds = []
    all_secondary_labels = []
    all_secondary_preds = []

    with tqdm(total=dataset_size, desc="Evaluating accuracy") as pbar:
        for inputs, (primary_labels, secondary_labels), dataset in dataloader:
            dataset = np.array(dataset)
            inputs = inputs.to(device)

            output1, output2 = model(inputs)
            _, preds1 = torch.max(output1, 1)
            _, preds2 = torch.max(output2, 1)

            preds1 = preds1.cpu()
            preds2 = preds2.cpu()

            all_secondary_labels.extend(secondary_labels.numpy())
            all_secondary_preds.extend(preds2.numpy())
            all_primary_labels.extend(primary_labels.numpy())
            all_primary_preds.extend(preds1.numpy())

            for ds in np.unique(dataset):
                dsidx = ds == dataset
                for preds, labels, class_counts, classnames, class_correct in zip(
                    [preds1, preds2],
                    [primary_labels, secondary_labels],
                    [primary_class_counts, secondary_class_counts],
                    [primary_classnames, secondary_classnames],
                    [primary_class_correct, secondary_class_correct],
                ):
                    pred_ds = preds[dsidx]
                    labels_ds = labels[dsidx]
                    correct_pred = (pred_ds == labels_ds).float()
                    class_counts[ds] += torch.bincount(
                        labels_ds, minlength=len(classnames)
                    )
                    class_correcti = torch.zeros(len(classnames)).to(preds.device)

                    for i in range(len(classnames)):
                        class_correcti[i] = correct_pred[labels_ds == i].sum()

                    class_correct[ds] += class_correcti

            pbar.update(dataloader.batch_size)

    primary_ncorrect, primary_ntotal, secondary_ncorrect, secondary_ntotal = (
        {},
        {},
        {},
        {},
    )
    for ncorrect, ntotal, class_counts, class_correct in zip(
        [primary_ncorrect, secondary_ncorrect],
        [primary_ntotal, secondary_ntotal],
        [primary_class_counts, secondary_class_counts],
        [primary_class_correct, secondary_class_correct],
    ):
        for ds in class_correct:
            ncorrect[ds] = class_correct[ds].numpy()
            ntotal[ds] = class_counts[ds].numpy()

    confusion_mat1 = confusion_matrix(
        all_primary_labels,
        all_primary_preds,  # normalize="true"
    )
    confusion_mat2 = confusion_matrix(
        all_secondary_labels,
        all_secondary_preds,  # normalize="true"
    )

    return (
        (primary_ncorrect, primary_ntotal),
        (secondary_ncorrect, secondary_ntotal),
        confusion_mat1,
        confusion_mat2,
    )


def test_image(data_dir: Path, model_path: Path, save_path: Path) -> None:
    """
    Run evaluation on all test data.

    Args:
        data_dir (str): Directory containing the data.
        model_path (str): Path to the trained model file.
    Returns:
        None
    """
    prep = PredictionPreparation(model_path, data_dir, save_path=save_path)
    model = prep.model
    dataloaders = prep.dataloaders["test"]
    primary_classnames = prep.primary_classnames
    secondary_classnames = prep.secondary_classnames
    save_path = prep.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    print(save_path)
    dataset_sizes = len(dataloaders.dataset)

    (
        (primary_ncorrect, primary_ntotal),
        (secondary_ncorrect, secondary_ntotal),
        cm1,
        cm2,
    ) = _count_class_correct_test_images(
        model, dataloaders, dataset_sizes, primary_classnames, secondary_classnames
    )
    _all_accuracy1, ds_accuracy1, class_accuracy1, accuracy1 = calculate_accuracy(
        primary_ncorrect, primary_ntotal, primary_classnames
    )
    _all_accuracy2, ds_accuracy2, class_accuracy2, accuracy2 = calculate_accuracy(
        secondary_ncorrect, secondary_ntotal, secondary_classnames
    )
    log.info(f"Dataset material accuracy:")
    log.info(ds_accuracy1)
    log.info(f"Dataset category accuracy:")
    log.info(ds_accuracy2)
    log.info(f"Class material accuracy:")
    log.info(class_accuracy1)
    log.info(f"Class category accuracy:")
    log.info(class_accuracy2)
    log.info(f"Overall material accuracy: {accuracy1:.2f}%")
    log.info(f"Overall category accuracy: {accuracy2:.2f}%")
    ds_accuracy1.to_csv(save_path / "primary_image_ds_accuracy.csv")
    class_accuracy1.to_csv(save_path / "primary_image_class_accuracy.csv")
    np.array([accuracy1]).tofile(save_path / "primary_image_accuracy.npy")
    ds_accuracy2.to_csv(save_path / "secondary_image_ds_accuracy.csv")
    class_accuracy2.to_csv(save_path / "secondary_image_class_accuracy.csv")
    np.array([accuracy2]).tofile(save_path / "secondary_image_accuracy.npy")
    cm1.tofile(save_path / "primary_confusion_matrix.npy")
    cm2.tofile(save_path / "secondary_confusion_matrix.npy")
    _save_confusion_matrix_image(
        cm2, secondary_classnames, save_path / "secondary_confusion_matrix.png"
    )
    _save_confusion_matrix_image(
        cm1, primary_classnames, save_path / "primary_confusion_matrix.png"
    )


def _save_confusion_matrix_image(
    confusion_matrix, classnames, save_path: Path, title="Confusion Matrix"
):
    """
    Generates and saves an image of the confusion matrix using matplotlib.

    Args:
        confusion_matrix (np.ndarray): The confusion matrix to visualize.
        classnames (list[str]): The class names corresponding to the confusion matrix.
        save_path (Path): The path to save the confusion matrix image.
        title (str): The title of the confusion matrix plot.
    """
    plt.figure(figsize=(10, 8))

    # Normalize each row by its sum (true class totals)
    normalized_cm = confusion_matrix.astype("float")
    row_sums = normalized_cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    normalized_cm = normalized_cm / row_sums

    plt.imshow(
        normalized_cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1
    )
    plt.title(title)
    plt.colorbar(label="Proportion of True Class")

    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=45, ha="right")
    plt.yticks(tick_marks, classnames)

    # Annotate the confusion matrix with original values and percentages
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            # Show both raw count and percentage
            raw_count = confusion_matrix[i, j]
            percentage = normalized_cm[i, j] * 100
            if percentage >= 3.0:  # Only show text if percentage is 3% or higher
                plt.text(
                    j,
                    i,
                    f"{raw_count}\n({percentage:.1f}%)",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=("white" if normalized_cm[i, j] > 0.5 else "black"),
                )

    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Confusion matrix image saved to {save_path}")
