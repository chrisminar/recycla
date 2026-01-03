from pathlib import Path
from typing import Final

import click
import numpy as np
import pandas as pd

from recycla import log
from recycla.classify.classification_utils import check_if_dir_is_from_root

POSSIBLE_PREFIXES: Final[list[str]] = ["image"]
POSSIBLE_SUFFIXES: Final[list[str]] = [
    "accuracy.npy",
    "ds_accuracy.csv",
    "class_accuracy.csv",
]
SUFFIXES: Final[list[str]] = [suffix.rsplit(".", 1)[0] for suffix in POSSIBLE_SUFFIXES]


def _compare_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Compare two pandas DataFrames and return a DataFrame highlighting the differences.
    If the DataFrames are equal, the returned DataFrame will contain the same values
    from both input DataFrames along with a column indicating no difference (Δ = 0).
    If the DataFrames are not equal, the returned DataFrame will show the differences
    between the two DataFrames, including the values from both DataFrames and the
    difference (Δ) between them.
    Parameters:
    df1 (pd.DataFrame): The first DataFrame to compare.
    df2 (pd.DataFrame): The second DataFrame to compare.
    Returns:
    pd.DataFrame: A DataFrame showing the comparison results, including the values
                  from both input DataFrames and the differences between them.
    """
    if df1.equals(df2):
        compare_df = pd.DataFrame(
            index=df1.index,
            columns=pd.MultiIndex.from_product([df1.columns, ["Model1", "Model2"]]),
        )
        compare_df[("Accuracy", "Model1")] = df1
        compare_df[("Accuracy", "Model2")] = df2
        compare_df[("Accuracy", "Δ")] = 0
    else:
        common_indices = df1.index.intersection(df2.index)
        df1 = df1.loc[common_indices]
        df2 = df2.loc[common_indices]
        compare_df = df1.compare(
            df2, keep_equal=True, keep_shape=True, result_names=("Model1", "Model2")
        )
        compare_df[("Accuracy", "Δ")] = (
            compare_df["Accuracy"]["Model2"] - compare_df["Accuracy"]["Model1"]
        )
    return compare_df


def _load_results(save_path: Path) -> dict:
    """Retrieve evaluation results from specified directory.

    This function reads accuracy and dataset-specific accuracy results from
    files in the given directory and organizes them into a dictionary.
    Args:
        save_path (Path): The directory path where the result files are stored.
    Returns:
        dict: A dictionary containing the results. The keys are prefixes, and
              the values are dictionaries with keys "accuracy", "ds_accuracy",
              and "class_accuracy" corresponding to the respective data.
    """
    results = {prefix: {} for prefix in POSSIBLE_PREFIXES}
    for prefix in POSSIBLE_PREFIXES:
        for suffix in POSSIBLE_SUFFIXES:
            path = save_path / f"secondary_{prefix}_{suffix}"
            if path.exists():
                if suffix == "accuracy.npy":
                    results[prefix]["accuracy"] = np.fromfile(path)
                elif suffix == "ds_accuracy.csv":
                    results[prefix]["ds_accuracy"] = pd.read_csv(path).set_index(
                        "Dataset"
                    )
                    results[prefix]["ds_accuracy"].columns = ["Accuracy"]
                elif suffix == "class_accuracy.csv":
                    results[prefix]["class_accuracy"] = pd.read_csv(path).set_index(
                        "Class"
                    )
                    results[prefix]["class_accuracy"].columns = ["Accuracy"]
    return results


def _compare_all_dataframes(results1: dict, results2: dict) -> dict:
    """
    Compare dataframes from two sets of results and return a dictionary of comparisons.
    This function takes two dictionaries of results, each containing dataframes
    identified by a prefix and suffix. It compares the dataframes with matching
    prefixes and suffixes from both dictionaries and stores the comparison results
    in a new dictionary.
    Args:
        results1 (dict): The first dictionary of results, where keys are prefixes
                         and values are dictionaries with suffixes as keys and
                         dataframes as values.
        results2 (dict): The second dictionary of results, with the same structure
                         as results1.
    Returns:
        dict: A dictionary containing the comparison results. The keys are prefixes
              and the values are dictionaries with suffixes as keys and comparison
              results as values.
    Notes:
        - The function assumes that `possible_prefixes` and `suffixes` are defined
          globally and accessible within the function.
        - If a dataframe with a specific prefix and suffix is not found in either
          results1 or results2, the function logs an informational message and skips
          the comparison for that prefix-suffix pair.
    """
    compares = {prefix: {} for prefix in POSSIBLE_PREFIXES}
    for prefix in POSSIBLE_PREFIXES:
        for suffix in SUFFIXES[1:]:
            result_in_1 = prefix in results1 and suffix in results1[prefix]
            result_in_2 = prefix in results2 and suffix in results2[prefix]
            if result_in_1 and result_in_2:
                compares[prefix][suffix] = _compare_df(
                    results1[prefix][suffix], results2[prefix][suffix]
                )
            else:
                log.info(
                    f"Results for {prefix} {suffix} found in results1 ({result_in_1}), results2({result_in_2}), skipping."
                )
    return compares


def _report_all_comparisons(stylized: dict, results1: dict, results2: dict) -> None:
    """
    Generates and logs a report comparing the accuracy metrics of two models.
    Args:
        stylized (dict): A dictionary containing stylized accuracy metrics for datasets and classes.
        results1 (dict): A dictionary containing the accuracy metrics of the first model.
        results2 (dict): A dictionary containing the accuracy metrics of the second model.
    Returns:
        None
    """

    for prefix in POSSIBLE_PREFIXES:
        if "accuracy" in results1[prefix] and "accuracy" in results2[prefix]:
            delta = results2[prefix]["accuracy"] - results1[prefix]["accuracy"]
            log.info(f"Overall {prefix} accuracies:")
            s = f"Model 1 -> Model 2: {results1[prefix]['accuracy'][0]:.2f}% ->"
            s += f" {results2[prefix]['accuracy'][0]:.2f}%"
            s += f"... {delta[0]:.2f}% {'improvement' if delta[0] > 0 else 'worse'}."
            log.info(s)
            print("")
            print("")
            print("")
        if "ds_accuracy" in stylized[prefix]:
            log.info(f"Dataset {prefix} accuracies:")
            log.info(stylized[prefix]["ds_accuracy"])
            print("")
            print("")
            print("")
        if "class_accuracy" in stylized[prefix]:
            log.info(f"Class {prefix} accuracies:")
            log.info(stylized[prefix]["class_accuracy"])
            print("")
            print("")
            print("")
        print("")
        print("")
        print("")


def _compare(results1_path: Path, results2_path: Path) -> None:
    results1_path = check_if_dir_is_from_root(results1_path)
    results2_path = check_if_dir_is_from_root(results2_path)
    results1 = _load_results(results1_path)
    results2 = _load_results(results2_path)
    compares = _compare_all_dataframes(results1, results2)
    _report_all_comparisons(compares, results1, results2)


@click.command()
@click.argument("results1_path", type=click.Path(path_type=Path))
@click.argument("results2_path", type=click.Path(path_type=Path))
def compare(results1_path: Path, results2_path: Path):
    """Compare results from two models."""
    _compare(results1_path, results2_path)
