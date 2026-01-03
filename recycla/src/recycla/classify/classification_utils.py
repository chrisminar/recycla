from pathlib import Path

import torch
from torch import Tensor

from recycla import ROOT_PATH, device
from recycla.process_data.io import import_classnames
from recycla.train.loader_utils import data_load


def check_if_dir_is_from_root(data_dir: Path | None) -> Path:
    """
    Checks if the data directory is from the root directory.
    Args:
        data_dir (Path): The data directory to check.
    Returns:
    """
    if data_dir is not None and not data_dir.is_absolute():
        data_dir = ROOT_PATH / data_dir
    if data_dir is not None and not data_dir.exists():
        raise FileNotFoundError(f"Directory not found at {data_dir}")
    return data_dir


class PredictionPreparation:
    def __init__(
        self,
        model_path: Path | None = None,
        data_dir: Path | None = None,
        file_path: Path | None = None,
        save_path: Path | None = None,
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.file_path = file_path
        self.save_path = save_path
        self.model = None
        self.dataloaders = None
        self.primary_classnames = None
        self.secondary_classnames = None
        self._model_path_available = False
        self._data_dir_available = False
        self._file_path_available = False
        self._save_path_available = False
        self._update_paths()
        self._load_classnames()
        self._dataloaders()
        self._load_model()

    def _update_paths(self):
        for attr in ["model_path", "data_dir", "file_path", "save_path"]:
            try:
                setattr(self, attr, check_if_dir_is_from_root(getattr(self, attr)))
                if getattr(self, attr) is not None:
                    setattr(self, f"_{attr}_available", True)
                else:
                    raise FileNotFoundError
                print(f"{attr} found at {getattr(self, attr)}")
            except FileNotFoundError as e:
                print(attr + " not found: " + str(e))
                setattr(self, f"_{attr}_available", False)

    def _load_classnames(self) -> None:
        primary_classnames, secondary_classnames = import_classnames()
        self.primary_classnames = primary_classnames
        self.secondary_classnames = secondary_classnames

    def _dataloaders(self) -> None:
        if not self._data_dir_available:
            return
        self.dataloaders, _data_sizes, primary_classnames, secondary_classnames = (
            data_load(self.data_dir, ["test"], None)
        )
        if self.primary_classnames is None:
            self.primary_classnames = primary_classnames
        if self.secondary_classnames is None:
            self.secondary_classnames = secondary_classnames

    def _load_model(self) -> None:
        if not self._model_path_available or self.model_path is None:
            return
        model = torch.load(
            self.model_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        self.model = model.to(device)


def topn(
    probabilities: Tensor, classnames: list[str], n: int = 5
) -> tuple[list[str], list[float]]:
    """
    Selects the top N class names and their corresponding probabilities.
    Args:
        probabilities (torch.Tensor): A tensor of probabilities for each class.
        classnames (list[str]): A list of class names corresponding to the probabilities.
        n (int, optional): The number of top classes to return. Defaults to 5.
    Returns:
        tuple[list[str], list[float]]: A tuple containing a list of the top N class names and a list of their corresponding probabilities.

    Raises:
        ValueError: If n is greater than the number of class'
    """
    if n > len(classnames):
        raise ValueError("n cannot be greater than the number of classnames")
    top_prob, top_catid = torch.topk(probabilities, n)

    top_names = [[classnames[catid] for catid in catidrow] for catidrow in top_catid]
    return top_names, top_prob
