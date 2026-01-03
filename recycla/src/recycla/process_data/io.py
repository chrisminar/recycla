from pathlib import Path

import yaml


def import_file_hierarchy() -> dict[str, str]:
    """
    For each label that might be found in raw_labeled data, convert it to a label that we will actually train on.
    The output will be a dictionary with "original_label": "training_label"
    """
    file_path = Path(__file__).parent.parent / "config/label_hierarchy.yaml"
    with open(file_path, "r") as f:
        d = yaml.safe_load(f)
    output = {}
    for key, value in d.items():
        output[key] = key
        if value is not None:
            for item in value:
                if item in output:
                    raise ValueError(f"Duplicate key found: {item}")
                output[item] = key

    return output


def import_classnames() -> tuple[list[str], list[str]]:
    file_path = Path(__file__).parent.parent / "config/label_hierarchy.yaml"
    with open(file_path, "r") as f:
        d = yaml.safe_load(f)
    primary_classnames = set()
    secondary_classnames = []
    for key in d:
        secondary_classnames.append(key)
        if "," in key:
            primary_classname = key.split(",")[0].strip()
            primary_classnames.add(primary_classname)
        else:
            primary_classnames.add(key)
    primary_classnames = list(sorted(primary_classnames))
    secondary_classnames = sorted(secondary_classnames)
    return primary_classnames, secondary_classnames
