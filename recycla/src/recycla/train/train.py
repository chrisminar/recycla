import time

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from recycla import ROOT_PATH, device, log
from recycla.train.CustomMobileNetV2 import CustomMobileNetV2
from recycla.train.loader_utils import data_load
from recycla.train.ModelType import ModelType


def calculate_class_weights(
    dataloader: DataLoader, num_classes: int, label_level: int
) -> torch.Tensor:
    """
    Calculate class weights based on the number of samples in each class.
    Uses sklearn's 'balanced' approach: n_samples / (n_classes * class_count)

    Args:
        dataloader (DataLoader): The dataloader for the training dataset.
        num_classes (int): The total number of classes.
        label_level (int): Which label to use (0 for primary, 1 for secondary).

    Returns:
        torch.Tensor: A tensor containing the class weights.
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # Count the occurrences of each class in the dataset
    for _, (labels), _ in dataloader:
        labels = labels[label_level]
        for label in labels.numpy():
            class_counts[label] += 1

    # Check for classes with zero samples
    if np.any(class_counts == 0):
        log.warning(
            f"Found classes with zero samples: {np.where(class_counts == 0)[0]}"
        )
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero

    # Calculate total samples
    total_samples = np.sum(class_counts)

    # Use sklearn's 'balanced' formula: n_samples / (n_classes * class_count)
    class_weights = total_samples / (num_classes * class_counts)

    # Optional: Cap extreme weights to prevent instability
    max_weight = 1.0  # Adjust based on your data
    class_weights = np.minimum(class_weights, max_weight)

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def train_model(
    model: nn.Module,
    dataloaders: dict,
    dataset_sizes: dict,
    criterion_primary: nn.Module,
    criterion_secondary: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    num_epochs: int,
) -> tuple[
    list[float], list[float], list[float], list[float], list[float], list[float]
]:
    """
    Trains a given model using the provided dataloaders, criterion, optimizer, and scheduler.
    Args:
        model (nn.Module): The neural network model to be trained.
        dataloaders (dict): A dictionary containing the training and validation dataloaders.
        dataset_sizes (dict): A dictionary containing the sizes of the training and validation datasets.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for updating the model parameters.
        scheduler (lr_scheduler._LRScheduler): The learning rate scheduler.
        num_epochs (int): The number of epochs to train the model.
    Returns:
        tuple: (train_losses, train_primary_accuracies, train_secondary_accuracies, val_losses, val_primary_accuracies, val_secondary_accuracies)
    """
    start_time = time.time()
    best_loss = float("inf")
    epochs_no_improve = 0  # Counter for early stopping
    patience = 10  # Number of epochs to wait before stopping

    train_losses = []
    train_primary_accuracies = []
    train_secondary_accuracies = []
    val_losses = []
    val_primary_accuracies = []
    val_secondary_accuracies = []

    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch+1}/{num_epochs}")

        epoch_train_loss = None
        epoch_train_primary_acc = None
        epoch_train_secondary_acc = None
        epoch_val_loss = None
        epoch_val_primary_acc = None
        epoch_val_secondary_acc = None

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_primary = 0
            running_corrects_secondary = 0

            with tqdm(
                total=dataset_sizes[phase],
                leave=False,
                desc=f"Epoch {epoch+1} [{phase}]",
            ) as pbar:
                for inputs, (primary_labels, secondary_labels), _dataset in dataloaders[
                    phase
                ]:
                    inputs = inputs.to(device)
                    primary_labels = primary_labels.to(device)
                    secondary_labels = secondary_labels.to(device)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs_primary, outputs_secondary = model(inputs)

                        _, preds_primary = torch.max(outputs_primary, 1)
                        _, preds_secondary = torch.max(outputs_secondary, 1)

                        loss_primary = criterion_primary(
                            outputs_primary, primary_labels
                        )
                        loss_secondary = criterion_secondary(
                            outputs_secondary, secondary_labels
                        )
                        loss = loss_primary + loss_secondary

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects_primary += torch.sum(
                        preds_primary == primary_labels.data
                    )
                    running_corrects_secondary += torch.sum(
                        preds_secondary == secondary_labels.data
                    )
                    pbar.update(dataloaders[phase].batch_size)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_primary = running_corrects_primary.double() / dataset_sizes[phase]
            epoch_acc_secondary = (
                running_corrects_secondary.double() / dataset_sizes[phase]
            )

            log.info(
                f"{phase} Loss: {epoch_loss:.4f} Primary Acc: {epoch_acc_primary:.4f} Secondary Acc: {epoch_acc_secondary:.4f}"
            )

            if phase == "train":
                epoch_train_loss = epoch_loss
                epoch_train_primary_acc = epoch_acc_primary.item()
                epoch_train_secondary_acc = epoch_acc_secondary.item()
            else:
                epoch_val_loss = epoch_loss
                epoch_val_primary_acc = epoch_acc_primary.item()
                epoch_val_secondary_acc = epoch_acc_secondary.item()

            # Early stopping logic
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    log.info("New Best Validation Loss!")
                    # Create .models directory if it doesn't exist
                    models_dir = ROOT_PATH / ".models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model, models_dir / "best_candidate.pth")
                    torch.save(
                        model.state_dict(),
                        models_dir / "best_weights_candidate.pth",
                    )
                else:
                    epochs_no_improve += 1
                    log.info(
                        f"No improvement in validation loss for {epochs_no_improve} epoch(s)."
                    )
                    if epochs_no_improve >= patience:
                        log.info("Early stopping triggered.")
                        time_elapsed = time.time() - start_time
                        log.info(
                            f"Training stopped early after {epoch+1} epochs. Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
                        )
                        # Append up to this epoch
                        if epoch_train_loss is not None:
                            train_losses.append(epoch_train_loss)
                        if epoch_train_primary_acc is not None:
                            train_primary_accuracies.append(epoch_train_primary_acc)
                        if epoch_train_secondary_acc is not None:
                            train_secondary_accuracies.append(epoch_train_secondary_acc)
                        if epoch_val_loss is not None:
                            val_losses.append(epoch_val_loss)
                        if epoch_val_primary_acc is not None:
                            val_primary_accuracies.append(epoch_val_primary_acc)
                        if epoch_val_secondary_acc is not None:
                            val_secondary_accuracies.append(epoch_val_secondary_acc)
                        return (
                            train_losses,
                            train_primary_accuracies,
                            train_secondary_accuracies,
                            val_losses,
                            val_primary_accuracies,
                            val_secondary_accuracies,
                        )

        # Append after both phases
        if epoch_train_loss is not None:
            train_losses.append(epoch_train_loss)
        if epoch_train_primary_acc is not None:
            train_primary_accuracies.append(epoch_train_primary_acc)
        if epoch_train_secondary_acc is not None:
            train_secondary_accuracies.append(epoch_train_secondary_acc)
        if epoch_val_loss is not None:
            val_losses.append(epoch_val_loss)
        if epoch_val_primary_acc is not None:
            val_primary_accuracies.append(epoch_val_primary_acc)
        if epoch_val_secondary_acc is not None:
            val_secondary_accuracies.append(epoch_val_secondary_acc)

    time_elapsed = time.time() - start_time
    log.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    log.info(f"Best val Loss: {best_loss:.4f}")

    # Clean up dataloaders and free memory
    for phase in ["train", "val"]:
        if phase in dataloaders:
            del dataloaders[phase]
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return (
        train_losses,
        train_primary_accuracies,
        train_secondary_accuracies,
        val_losses,
        val_primary_accuracies,
        val_secondary_accuracies,
    )


def _train(
    data_dir,
    nepochs: int,
    checkpoint_path: str | None = None,
    weight_labels: bool = False,
    transform_kwargs: dict | None = None,
    freeze_backbone: bool = False,
    unfreeze_last_n_blocks: int = 0,
    model_type: ModelType = ModelType.MOBILENET_V2,
):
    n_params = ModelType.n_params(model_type)
    (
        dataloaders,
        dataset_sizes,
        primary_class_names,
        secondary_class_names,
    ) = data_load(
        data_dir, ["train", "val"], n_params=n_params, transform_kwargs=transform_kwargs
    )

    # Create model based on model_type
    model_class, model_kwargs = ModelType.get_model_class_and_kwargs(model_type)
    model = model_class(
        len(primary_class_names),
        len(secondary_class_names),
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        **model_kwargs,
    )
    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path:
        log.info(f"Loading checkpoint from {checkpoint_path}...")
        model_state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(model_state_dict)
        log.info("Checkpoint loaded successfully.")

    if weight_labels:
        # Calculate class weights for primary labels
        primary_class_weights = calculate_class_weights(
            dataloaders["train"], len(primary_class_names), 0
        )
        secondary_class_weights = calculate_class_weights(
            dataloaders["train"], len(secondary_class_names), 1
        )
        for i, class_name in enumerate(primary_class_names):
            log.info(
                f"Primary class '{class_name}' weight: {primary_class_weights[i].item():.4f}"
            )
        for i, class_name in enumerate(secondary_class_names):
            log.info(
                f"Secondary class '{class_name}' weight: {secondary_class_weights[i].item():.4f}"
            )

        # Define loss functions with class weights
        criterion_primary = nn.CrossEntropyLoss(weight=primary_class_weights)
        criterion_secondary = nn.CrossEntropyLoss(weight=secondary_class_weights)
    else:
        criterion_primary = nn.CrossEntropyLoss()
        criterion_secondary = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # epoch lr * 0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion_primary,
        criterion_secondary,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=nepochs,
    )


@click.command()
@click.option(
    "--data-dir", default="data/labeled_data", help="Directory with training data"
)
@click.option("--nepochs", default=20, help="Number of epochs to train.")
@click.option(
    "--checkpoint-path",
    default=None,
    # Note: I was getting worse convergence while loading from a checkpoint
    # default=".models/best_weights_checkpoint.pth",
    help="Path to a checkpoint file to resume training from.",
)
@click.option(
    "-w",
    "--weight-labels",
    is_flag=True,
    default=False,
    help="Use class weights to handle imbalanced datasets.",
)
@click.option(
    "--freeze-backbone",
    is_flag=True,
    default=False,
    help="Freeze the backbone feature extractor.",
)
@click.option(
    "--unfreeze-last-n-blocks",
    default=2,
    help="Number of last inverted residual blocks to unfreeze (when backbone is frozen).",
)
@click.option(
    "--model-type",
    type=click.Choice(
        [model_type.value for model_type in ModelType], case_sensitive=False
    ),
    default=ModelType.EFFICIENTNET_V2_L.value,
    help="Type of neural network model to use.",
)
def train(
    data_dir,
    nepochs,
    checkpoint_path,
    weight_labels,
    freeze_backbone,
    unfreeze_last_n_blocks,
    model_type,
):
    """
    Train a model using the provided data directory, number of epochs, and optional checkpoint.
    """
    # Convert string to ModelType enum
    model_type_enum = ModelType.from_string(model_type)

    (
        train_losses,
        train_primary_accuracies,
        train_secondary_accuracies,
        val_losses,
        val_primary_accuracies,
        val_secondary_accuracies,
    ) = _train(
        data_dir,
        nepochs,
        checkpoint_path,
        weight_labels,
        None,
        freeze_backbone,
        unfreeze_last_n_blocks,
        model_type_enum,
    )
    log.info("Train losses: %s", train_losses)
    log.info("Train primary accuracies: %s", train_primary_accuracies)
    log.info("Train secondary accuracies: %s", train_secondary_accuracies)
    log.info("Val losses: %s", val_losses)
    log.info("Val primary accuracies: %s", val_primary_accuracies)
    log.info("Val secondary accuracies: %s", val_secondary_accuracies)


if __name__ == "__main__":
    data_dir = "data/labeled_data"
    nepochs = 20
    checkpoint_path = None
    weight_labels = False
    freeze_backbone = False
    unfreeze_last_n_blocks = 2
    model_type_enum = ModelType.EFFICIENTNET_V2_L

    _train(
        data_dir=data_dir,
        nepochs=nepochs,
        checkpoint_path=checkpoint_path,
        weight_labels=weight_labels,
        transform_kwargs=None,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        model_type=model_type_enum,
    )
