import numpy as np
import pytest
import torch
from PIL import Image

from recycla.train.loader_utils import data_load
from recycla.train.train import calculate_class_weights


@pytest.fixture
def mock_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Create a dummy image using numpy
    image_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    save = lambda image, parentdir, i: image.save(parentdir / f"dummy_image{i}.jpg")
    phases = ["test"]
    classes = ["metal, aluminum", "metal, generic", "plastic, generic"]
    for phase in phases:
        (data_dir / phase).mkdir(exist_ok=True)
        metadata_path = data_dir / phase / "metadata.csv"
        metadata_path.touch()
        for i, cls in enumerate(classes):
            (data_dir / phase / cls).mkdir(parents=True)
            for _ in range(i + 2):
                save(image, data_dir / phase / cls, _)
                with open(metadata_path, "a") as f:
                    f.write(f"dummy_image.jpg,{cls},test\n")
    return data_dir


def test_calculate_class_weights(mock_data_dir):
    dataloaders, dataset_sizes, primary_classes, secondary_classes = data_load(
        data_dir=mock_data_dir,
        phases=["test"],
    )

    # Mock device to avoid GPU dependency
    global device
    device = torch.device("cpu")

    # Test with secondary classes (label_level=1)
    # Class counts from fixture: [2, 3, 4] for the 3 secondary classes
    class_weights = calculate_class_weights(dataloaders["test"], 3, 1).to(device)

    # Expected calculation using sklearn's 'balanced' approach:
    # total_samples = 2 + 3 + 4 = 9
    # class_weights = total_samples / (num_classes * class_counts)
    # class_weights = 9 / (3 * [2, 3, 4]) = [1.5, 1.0, 0.75]
    # After capping at 1.0: [1.0, 1.0, 0.75]
    expected_weights = torch.tensor([1.0, 1.0, 0.75], dtype=torch.float32).to(device)

    assert torch.allclose(
        class_weights, expected_weights, atol=1e-4
    ), f"Class weights are incorrect. Got {class_weights}, expected {expected_weights}"
