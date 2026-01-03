from unittest import mock

import numpy as np
import pytest
from PIL import Image
from torch import Tensor

from recycla.train.loader_utils import RecyclaDataset, data_load


@pytest.fixture
def mock_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Create a dummy image using numpy
    image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    save = lambda image, parentdir: image.save(parentdir / "dummy_image.jpg")
    phases = ["train", "val", "test"]
    classes = ["class1", "class2"]
    for phase in phases:
        for cls in classes:
            (data_dir / phase / cls).mkdir(parents=True)
            save(image, data_dir / phase / cls)
    return data_dir


def test_recycla_dataset_len(mock_data_dir):
    dataset = RecyclaDataset(data_dir=mock_data_dir, phase="train")
    assert len(dataset) == 2


def test_recycla_dataset_getitem(mock_data_dir):
    dataset = RecyclaDataset(data_dir=mock_data_dir, phase="train")
    dataset.metadata = mock.MagicMock()
    dataset.metadata.__getitem__.side_effect = lambda idx: [0, 1]
    assert isinstance(dataset[0], tuple)
    assert len(dataset[0]) == 3
    assert isinstance(dataset[0][0], Tensor)
    assert isinstance(dataset[0][1], tuple)
    assert isinstance(dataset[0][1][0], int)
    assert isinstance(dataset[0][1][1], int)


def test_data_load(mock_data_dir):
    phases = ["train", "val", "test"]
    dataloaders, dataset_sizes, primary_classes, secondary_classes = data_load(
        data_dir=mock_data_dir, phases=phases
    )
    assert set(dataloaders.keys()) == set(phases)
    assert set(dataset_sizes.keys()) == set(phases)
    assert np.all(primary_classes == np.array(["class1", "class2"]))
    assert np.all(secondary_classes == np.array(["class1", "class2"]))
