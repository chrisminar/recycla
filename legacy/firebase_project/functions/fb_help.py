from datetime import datetime
from pathlib import Path

from firebase_admin import credentials, firestore, initialize_app

try:
    service_account_key = (
        Path(__file__).parents[2] / ".secrets" / "service_account_key.json"
    )
    cred = credentials.Certificate(service_account_key)
    initialize_app(
        cred, {"databaseURL": "https://recyclo-c0fd1-default-rtdb.firebaseio.com/"}
    )
    db = firestore.client()
except Exception as e:
    cred = credentials.ApplicationDefault()
    initialize_app(
        cred, {"databaseURL": "https://recyclo-c0fd1-default-rtdb.firebaseio.com/"}
    )
    db = firestore.client()

secondary_label_map = {
    "glass": "glass",
    "glass, bottle": "glass",
    "glass, container": "glass",
    "glass, wine_bottle": "glass, wine_bottle",
    "metal": "metal",
    "metal, aluminum_aerosol": "metal",
    "metal, steel_aerosol": "metal",
    "metal, aluminum_can": "metal, aluminum_can",
    "metal, aluminum_foil": "metal, aluminum_foil",
    "metal, steel_can": "metal, steel_can",
    "paper": "paper",
    "paper, mail": "paper",
    "paper, paperback_book": "paper",
    "paper, shredded_paper": "paper",
    "paper, paper_bag": "paper",
    "paper, corrected_cardboard": "paper, corrugated_cardboard",
    "paper, pizza_box": "paper, corrugated_cardboard",
    "paper, egg_carton": "paper, egg_carton",
    "paper, paperboard": "paper, paperboard",
    "paper, paper_cup": "paper, paper_cup",
    "paper, paper_towel": "paper, paper_towel",
    "plastic": "plastic",
    "plastic, utensil": "plastic",
    "plastic, tub": "plastic",
    "plastic, bag": "plastic, film",
    "plastic, bottle": "plastic, bottle",
    "plastic, packaging": "plastic, film",
    "mixed, tetra_pak": "mixed, tetra_pak",
    "waste": "waste",
    "waste, styrofoam": "waste",
    "compost": "compost",
    "mixed, spiral_wound": "mixed",
    "miscellaneous": "miscellaneous",
    "miscellaneous, background": "miscellaneous, background",
    "miscellaneous_test": "miscellaneous",
}


def is_interesting_to_read(sub_material: str) -> bool:
    return sub_material in set(
        [
            "glass",
            "glass, wine_bottle",
            "metal",
            "metal, aluminum_can",
            "metal, steel_can",
            "paper, egg_carton",
            "paper, paperboard",
            "paper, paper_cup",
            "plastic",
            "plastic, bottle",
            "mixed, tetra_pak",
        ]
    )


def _check_prediction(label_pred: str | None, label: str | None) -> bool:
    """
    Determine if a prediction should be considered based on the presence of prediction and ground truth labels.

    Args:
        label_pred (str | None): The predicted label, or None if not available.
        label (str | None): The ground truth label, or None if not available.

    Returns:
        bool | None: True if there is not enough information to consider the prediction (both are None),
            False if there is ground truth or prediction exists, None otherwise.
    """
    if label_pred is None:  # no prediction
        if label is None:  # also no ground truth, don't use
            return True
        else:
            return False  # ground truth
    return False  # prediction exists


def _check_ignore_list(
    label: str | None, label_pred: str | None, ignore_list: list[str]
) -> bool:
    """
    Check if a label or predicted label is in the ignore list.

    Args:
        label (str | None): The ground truth label, or None if not available.
        label_pred (str | None): The predicted label, or None if not available.
        ignore_list (list[str]): List of labels to ignore.

    Returns:
        bool: True if the label or predicted label should be ignored, False otherwise.
    """
    if (label in ignore_list) or (label is None and label_pred in ignore_list):
        return True
    return False


def video_is_bounty_valid(vid: dict) -> int:
    """Check if a video is valid for bounty based on its class ID."""
    vid_gt = vid.get("user_ground_truth")
    vid_cls = vid.get("class_id")

    ignore_list = [
        "miscellaneous, test",
        "miscellaneous",
        "miscellaneous, background",
    ]

    # bounty is valid if it has a class and a ground truth
    # does not have to match

    # missing either classification -> always 0
    if vid_gt is None or vid_cls is None:
        return 0

    if _check_ignore_list(
        vid_gt.get("sub_material"), vid_cls.get("sub_material"), ignore_list
    ):
        return 0

    return 1


def is_primary_prediction_correct(
    user_gt: dict | None, predicted_cls: dict | None
) -> bool | None:
    key = "material"
    ignore_list = ["miscellaneous"]
    return is_prediction_correct(user_gt, predicted_cls, key, ignore_list, None)


def is_secondary_prediction_correct(
    user_gt: dict | None, predicted_cls: dict | None
) -> bool | None:
    key = "sub_material"
    ignore_list = [
        "miscellaneous",
        "miscellaneous, test",
        "miscellaneous, background",
    ]
    label_map = secondary_label_map
    return is_prediction_correct(user_gt, predicted_cls, key, ignore_list, label_map)


def is_prediction_correct(
    user_gt: dict | None,
    predicted_cls: dict | None,
    key: str,
    ignore_list: list[str],
    label_map: dict[str, str],
) -> bool | None:
    """
    Check if the primary label is correct based on user ground truth and predicted class.

    Both dicts should have these keys
    material: str
    sub_material: str

    Args:
        user_gt (dict): User ground truth dictionary.
        predicted_cls (dict): Predicted class dictionary.

    Returns:
        bool: True if the primary label is correct, False otherwise. None if it should be ignored.
    """
    label = user_gt.get(key) if user_gt else None
    label_pred = predicted_cls.get(key) if predicted_cls else None
    if _check_prediction(label, label_pred):
        return

    # ignore some labels?
    if _check_ignore_list(label, label_pred, ignore_list):
        return

    # map label to the correct label
    if label_map is not None:
        if label in label_map:
            label = label_map[label]
        if label_pred in label_map:
            label_pred = label_map[label_pred]

    # correct prediction
    if label == label_pred:
        return True
    # no gt, assume correct
    if label is None:
        return True
    # wrong prediction
    if label != label_pred:
        return False
    # no data, do nothing
    return None


def get_materials(vid: dict) -> tuple[str | None, str | None]:
    """Get material and sub_material from video dict.
    Get from user ground truth if it exists, otherwise get from class id.
    """
    vid_gt = vid.get("user_ground_truth")
    vid_cls = vid.get("class_id")

    try:
        material = vid_gt["material"]
    except (KeyError, TypeError):
        # if vid_gt is none type error
        # if vid_gt is dict but no material key, key error. In either case use vid_cls as backup
        material = vid_cls.get("material")
    try:
        sub_material = vid_gt["sub_material"]
    except (KeyError, TypeError):
        sub_material = vid_cls.get("sub_material")
    if sub_material is not None:
        sub_material = secondary_label_map.get(sub_material, sub_material)
    return material, sub_material


def date_2_month_string(date: datetime) -> str:
    """
    Convert a datetime object to a string in the format 'YYYY-MM'.

    Args:
        date (datetime): The datetime object to convert.

    Returns:
        str: The formatted string.
    """
    if date is None:
        return None
    return date.strftime("%m-%Y")


def date_2_week_string(date: datetime) -> str:
    """
    Convert a datetime object to a string in the format 'YYYY-WW'.

    Args:
        date (datetime): The datetime object to convert.

    Returns:
        str: The formatted string.
    """
    if date is None:
        return None
    return date.strftime("%Y-W%U")  # %U is the week number of the year (00-53)


def get_week_number(date: datetime) -> int:
    """
    Get the week number of the year (Sunday as the first day, week 0 for days before the first Sunday),
    matching strftime('%U').
    """
    return int(date.strftime("%U"))


def get_videos(pi_id: str):
    # get pi document
    pi = db.collection("pis").document(pi_id).get().to_dict()

    # get all video docs from that pi
    videos = []
    for doc_id in pi["videos"]:
        doc = db.collection("videos").document(doc_id).get()
        if doc.exists:
            videos.append(doc.to_dict())
    return videos


def count_documents(collection_name: str) -> int:
    """
    Returns the number of documents in the 'videos' collection using an aggregation query.

    Returns:
        int: The total number of documents in the 'videos' collection.
    """
    videos_ref = db.collection(collection_name)
    count_query = videos_ref.count()  # Use the count aggregation
    count_result = count_query.get()
    return count_result[0][0].value  # The count is stored in the 'value' attribute
