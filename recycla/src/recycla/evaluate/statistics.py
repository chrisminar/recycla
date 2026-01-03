import numpy as np
import pandas as pd


def calculate_accuracy(
    ncorrect: dict[str, np.ndarray],
    ntotal: dict[str, np.ndarray],
    classnames: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """Calculate the accuracy of classifications.

    Args:
        ncorrect (dict[str, np.ndarray]): A dictionary where keys are dataset names and values are arrays of correct predictions per class.
        ntotal (dict[str, np.ndarray]): A dictionary where keys are dataset names and values are arrays of total samples per class.
        classnames (list[str]): A list of class names corresponding to the arrays in ncorrect and ntotal.
    Returns:
        tuple[float, pd.Series, pd.Series, float]:
            - all_accuracy (float): Overall accuracy across all datasets and classes.
            - ds_accuracy (pd.Series): Accuracy per dataset.
            - class_accuracy (pd.Series): Accuracy per class.
            - accuracy (float): Redundant overall accuracy (same as all_accuracy).
    """
    df = pd.DataFrame.from_dict(ncorrect, orient="index")
    df.columns = classnames
    df = df.stack().reset_index()
    df.columns = ["Dataset", "Class", "correct"]
    df = df.set_index(["Class", "Dataset"])

    samples_df = pd.DataFrame.from_dict(ntotal, orient="index")
    samples_df.columns = classnames
    samples_df = samples_df.stack().reset_index()
    samples_df.columns = ["Dataset", "Class", "samples"]
    samples_df = samples_df.set_index(["Class", "Dataset"])

    df_all = df.join(samples_df["samples"])

    df_ds_agg = df_all.groupby(level=1).sum()
    df_class_agg = df_all.groupby(level=0).sum()
    df_agg = df_all.groupby(level=0).sum().sum()

    all_accuracy = 100 * df_all["correct"] / df_all["samples"]
    ds_accuracy = 100 * df_ds_agg["correct"] / df_ds_agg["samples"]
    class_accuracy = 100 * df_class_agg["correct"] / df_class_agg["samples"]
    accuracy = 100 * df_agg["correct"] / df_agg["samples"]

    return all_accuracy, ds_accuracy, class_accuracy, accuracy
