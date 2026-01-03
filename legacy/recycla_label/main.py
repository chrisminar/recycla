import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from firebase_admin import firestore
from firebase_admin.firestore import firestore as firestore_client

from recycla import DATA_PATH, ROOT_PATH, log
from recycla.classify.classification_utils import (
    PredictionPreparation,
    clense_input_images,
    topn,
)
from recycla.classify.classify import _classify_images, classify_sequence
from recycla.connect.firebase_models import DataLabel, FirebaseVideoModel
from recycla.connect.firebase_util import get_firebase_credentials
from recycla.process_data.io import (
    h264_file_to_buffer,
    import_classnames,
    parse_h264_file,
)
from recycla.vision.vision import prepare_pi_images


def get_videos(start_path: Path) -> List[Path]:
    """
    Recursively find all .h264 video files under the given start path.
    Raises FileNotFoundError if the path does not exist.
    """
    if not start_path.exists():
        raise FileNotFoundError(f"Path {start_path} does not exist.")
    videos = list(start_path.rglob("*.h264"))
    return videos


def sub_material_to_material(sub_material: str) -> str:
    """
    Map a sub-material string to its general material category.
    """
    if "glass" in sub_material:
        return "glass"
    if "metal" in sub_material:
        return "metal"
    if "paper" in sub_material:
        return "paper"
    if "plastic" in sub_material:
        return "plastic"
    if "mixed" in sub_material:
        return "mixed"
    if "waste" in sub_material:
        return "waste"
    if "compost" in sub_material:
        return "compost"
    return "miscellaneous"


def init_dict(video_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Initialize a dictionary of video metadata from a list of video paths.
    """
    return {
        video_path.stem: {
            "id": video_path.stem,
            "h264_path": video_path,
            "gt_sub": video_path.parent.stem,
            "gt_mat": sub_material_to_material(video_path.parent.stem),
        }
        for video_path in video_paths
    }


def setup_firebase() -> firestore_client.CollectionReference:
    """
    Set up and return a reference to the 'videos' collection in Firestore.
    """
    # setup firebase
    try:
        get_firebase_credentials()
    except:
        # firebase already setup, skip
        pass

    db = firestore.client()
    collection = db.collection("videos")
    return collection


def fetch_video_models(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Fetch video model metadata from Firestore and update the data dictionary in-place.
    """
    collection = setup_firebase()
    for video_key in data:
        reference = collection.document(video_key)
        videodict = reference.get().to_dict()
        class_id = videodict.get("class_id")
        class_id = DataLabel.from_dict(class_id) if class_id is not None else None
        user_ground_truth = (
            DataLabel.from_dict(videodict.get("user_ground_truth"))
            if videodict.get("user_ground_truth")
            else None
        )
        model = FirebaseVideoModel(
            id=videodict.get("id"),
            piid=videodict.get("piid"),
            video_gcs_path=videodict.get("video_gcs_path"),
            preview_gcs_path=videodict.get("preview_gcs_path"),
            upload_date=videodict.get("upload_date"),
            process_date=videodict.get("process_date"),
            class_id=class_id,
            confidence=videodict.get("confidence"),
            user_ground_truth=user_ground_truth,
            firmware_version=videodict.get("firmware_version"),
            report=videodict.get("report"),
            valid_frames=videodict.get("valid_frames"),
        )
        data[video_key].update({"ref": reference, "model": model})


def update_ground_truth(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Update the ground truth in the data dictionary based on the model's class ID.
    """
    for key in data:
        model = data[key]["model"]
        if model.user_ground_truth is not None:
            data[key]["gt_sub"] = model.user_ground_truth.sub_material
            data[key]["gt_mat"] = model.user_ground_truth.material
        elif model.class_id is not None:
            data[key]["gt_sub"] = model.class_id.sub_material
            data[key]["gt_mat"] = model.class_id.material
        else:
            # don't update, hope it works :)
            pass


def create_raw_label_data_dir(
    dir_name: str,
    data_path_root: Path = DATA_PATH / "raw_labeled_data",
) -> None:
    """
    Create a directory for raw labeled data based on the model path and class names.
    """
    _, second_class = import_classnames()
    base_dir = data_path_root / dir_name
    base_dir.mkdir(parents=True, exist_ok=True)
    for cls in second_class:
        (base_dir / cls).mkdir(parents=True, exist_ok=True)


def videos_to_images(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Parse each video's h264 file into images and store them in the data dictionary.
    """
    for key in data:
        video_path = data[key]["h264_path"]
        images = parse_h264_file(video_path)
        data[key]["images"] = images


def prep_model(
    model_path: Path = ROOT_PATH / ".models/best.pth",
) -> Tuple[Any, List[str], List[str]]:
    """
    Prepare and return the model and its class names for prediction.
    """
    prep = PredictionPreparation(
        model_path,
    )
    prep.model.eval()
    return prep.model, prep.primary_classnames, prep.secondary_classnames


def run_model_on_images(
    data: Dict[str, Dict[str, Any]],
    model: Any,
    primary: List[str],
    secondary: List[str],
) -> None:
    """
    Run the model on the images in the data dictionary and store probabilities.
    """
    for key in data:
        images = data[key]["images"]
        imgs_tensor = clense_input_images(images)
        primary_probabilities, secondary_probabilities = _classify_images(
            imgs_tensor, model, 64
        )
        primary_topnames, _ = topn(primary_probabilities, primary, 1)
        secondary_topnames, _ = topn(secondary_probabilities, secondary, 1)
        data[key]["primary_probabilities"] = [p[0] for p in primary_topnames]
        data[key]["secondary_probabilities"] = [p[0] for p in secondary_topnames]


def gt_index_match(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Mark indices in each video's predictions that match the ground truth sub-material.
    """
    for key in data:
        num_images = len(data[key]["images"])
        data_label = np.zeros(num_images, dtype=int)
        prediction = data[key]["secondary_probabilities"]
        gt = data[key]["gt_sub"]
        index = np.where(np.array(prediction) == gt)[0]
        data_label[index] = 1
        data[key]["gt_match_index"] = data_label


def _get_best_images(file_path: Path) -> np.ndarray:
    """
    Classify and return the indices of the best images from a video file.
    """
    buffer = h264_file_to_buffer(file_path)

    _, _, index = prepare_pi_images(buffer, 3)
    return index


def get_best_images(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Classify and store the best image indices for each video in the data dictionary.
    """
    log.setLevel("WARNING")
    for key in data:
        video_path = data[key]["h264_path"]
        index = _get_best_images(video_path)
        num_images = len(data[key]["images"])
        data_label = np.zeros(num_images, dtype=int)
        data_label[index] = 1
        data[key].update(
            {
                "preview_index": data_label,
            }
        )
    log.setLevel("INFO")


def _get_valid_frames(data: Dict[str, Dict[str, Any]], current_key: str) -> List[int]:
    """
    Return a list where each element is 0 if the frame is background, 1 otherwise.
    """
    secondary_probs = data[current_key]["secondary_probabilities"]
    if data[current_key]["model"].valid_frames is not None:
        vf = [0] * len(secondary_probs)
        for idx in data[current_key]["model"].valid_frames:
            vf[idx] = 1
        return vf
    return [0 if prob == "miscellaneous, background" else 1 for prob in secondary_probs]


def add_valid_frames(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Add a 'valid_frames' key to each video, indicating valid (1) or invalid (0) frames.
    """
    for key in data:
        data[key]["valid_frames"] = _get_valid_frames(data, key)
    for key in data:
        data[key]["most_recent_index_edited"] = None


def save_pickle(data: Dict[str, Dict[str, Any]], start_path: Path) -> None:
    """
    Save the data dictionary to a pickle file, removing unpickleable references.
    """
    for key in data:
        if "ref" in data[key]:
            del data[key]["ref"]
    output_path = ROOT_PATH / "recycla_label/data" / f"{start_path.name}_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def get_sub_materials(material: str, secondary: List[str]) -> List[str]:
    """
    Return a list of sub-materials from secondary that match the given material.
    """
    valid_subs = []
    for sub_mat in secondary:
        if material in sub_mat:
            valid_subs.append(sub_mat)
    return valid_subs


def get_pkl() -> Dict[str, Dict[str, Any]]:
    """
    Load and return a pickled data dictionary from a fixed path.
    """
    path = ROOT_PATH / "recycla_label/data/2025-05-01_2025-06-15_data.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_data_pkl(start_path: Path) -> Dict[str, Dict[str, Any]]:
    """Generate a data dictionary from video files in the given path, process them, and save the results."""
    video_paths = get_videos(start_path)  # get paths to files
    data = init_dict(video_paths)  # get ground truth
    fetch_video_models(data)  # fetch video models from firebase
    update_ground_truth(data)
    create_raw_label_data_dir(
        dir_name=start_path.name, data_path_root=ROOT_PATH / "recycla_label/data"
    )
    videos_to_images(data)
    model, primary, secondary = prep_model()  # prep model to run
    run_model_on_images(data, model, primary, secondary)  # run model on images
    gt_index_match(data)  # figure out which images match the gt
    get_best_images(data)  # figure out preview images
    add_valid_frames(data)
    # save_pickle(data, start_path)
    return data


def update_cloud_gt(data, current_key: str, gt: DataLabel):
    """
    Update the ground truth in Firestore for a specific video.
    """
    reference = data[current_key]["ref"]
    if not reference:
        collection = setup_firebase()
        reference = collection.document(current_key)

    reference.set(
        {
            "user_ground_truth": asdict(gt),
        },
        merge=True,
    )
    print(
        f"Updated ground truth for {current_key} to {gt.material}, {gt.sub_material}."
    )


def update_cloud_index(data, current_key: str):
    """
    Update the ground truth in Firestore for a specific video.
    """
    reference = data[current_key]["ref"]
    if not reference:
        collection = setup_firebase()
        reference = collection.document(current_key)

    valid_frames = np.argwhere(data[current_key]["valid_frames"]).squeeze().tolist()
    reference.set(
        {
            "valid_frames": valid_frames,
        },
        merge=True,
    )
    print(f"Updated valid frames for {current_key} in Firestore.")
