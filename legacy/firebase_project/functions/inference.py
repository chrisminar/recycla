"""
Video classification inference module for Firebase Cloud Functions.
Contains all the video processing, model loading, and classification logic.
"""

import io
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
from typing import BinaryIO

import numpy as np
import torch
from firebase_admin import firestore
from firebase_functions import firestore_fn
from google.api_core import exceptions as gcs_exceptions
from google.cloud import storage

from recycla import DEFAULT_BUCKET_NAME

# Import from recycla package (copied during deployment)
from recycla.classify.classify import pi_predict
from recycla.process_data.io import import_classnames
from recycla.vision.vision import prepare_pi_images

# Global variables for model and classnames (loaded during cold start)
MODEL = None
PRIMARY_CLASSNAMES = None
SECONDARY_CLASSNAMES = None


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_retry: tuple = (
        gcs_exceptions.ServiceUnavailable,
        gcs_exceptions.DeadlineExceeded,
        gcs_exceptions.InternalServerError,
        gcs_exceptions.TooManyRequests,
        ConnectionError,
        TimeoutError,
    ),
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which delay increases after each retry
        exceptions_to_retry: Tuple of exceptions that should trigger a retry
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions_to_retry as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts. Last error: {e}"
                        )
                        raise e

                    print(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                except Exception as e:
                    # Don't retry for non-transient errors
                    print(f"Non-retryable error in {func.__name__}: {e}")
                    raise e

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


@retry_with_backoff(max_retries=3, initial_delay=1.0)
def download_video_from_gcs(bucket_name: str, blob_name: str) -> BinaryIO:
    """Download video file from Google Cloud Storage with automatic retry."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Check if blob exists before attempting download
    # This will also be retried in case of transient failures
    if not blob.exists():
        raise FileNotFoundError(
            f"Video file {blob_name} not found in bucket {bucket_name}"
        )

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    return buffer


@retry_with_backoff(max_retries=3, initial_delay=2.0)
def load_model_from_gcs(
    bucket_name: str = DEFAULT_BUCKET_NAME, model_path: str = "models/best_cloud.pth"
):
    """Load PyTorch model from Google Cloud Storage with automatic retry."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Download model to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as temp_file:
            temp_model_path = temp_file.name
        blob.download_to_filename(temp_model_path)

        # Load model
        model = torch.load(temp_model_path, map_location="cpu", weights_only=False)
        model.eval()

        # Clean up temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def initialize_model_and_classnames():
    """Initialize model and classnames during cold start."""
    global MODEL, PRIMARY_CLASSNAMES, SECONDARY_CLASSNAMES

    try:
        print("Loading model and classnames during cold start...")

        # Load model
        MODEL = load_model_from_gcs()
        if MODEL is None:
            print("Failed to load model during initialization")
            return False

        # Load classnames
        PRIMARY_CLASSNAMES, SECONDARY_CLASSNAMES = import_classnames()

        print("Model and classnames loaded successfully")
        return True

    except Exception as e:
        print(f"Error during initialization: {e}")
        return False


def run_inference(event: firestore_fn.Event[firestore_fn.Change]) -> dict:
    """
    Triggers when a new document is added to the videos collection.
    Downloads the video, classifies it, and updates the document with the prediction.
    """
    global MODEL, PRIMARY_CLASSNAMES, SECONDARY_CLASSNAMES
    try:
        # Initialize model and classnames if not already loaded
        if MODEL is None or PRIMARY_CLASSNAMES is None or SECONDARY_CLASSNAMES is None:
            print("Model and classnames not loaded, initializing...")
            success = initialize_model_and_classnames()
            if not success:
                print("Failed to initialize model and classnames")
                return {
                    "status": "error",
                    "message": "Failed to initialize model and classnames",
                }

        # Get the video document data
        video_data = event.data.to_dict()
        video_id = event.params["id"]

        print(f"Processing new video: {video_id}")

        # Get video path from document
        video_gcs_path = video_data.get("video_gcs_path")
        if not video_gcs_path:
            print("No video_gcs_path found in document")
            return {"status": "error", "message": "No video_gcs_path found"}

        # Download video from GCS
        print(f"Downloading video from: {video_gcs_path}")
        video_buffer = download_video_from_gcs(DEFAULT_BUCKET_NAME, video_gcs_path)

        # Prepare images for classification
        preview_images, _, _ = prepare_pi_images(video_buffer, n=3)

        if not preview_images:
            print("No images extracted from video")
            return {"status": "error", "message": "No images extracted from video"}

        # Run classification
        prediction, confidence = pi_predict(
            MODEL,
            PRIMARY_CLASSNAMES,
            SECONDARY_CLASSNAMES,
            preview_images,
            batchsize=3,
        )

        if prediction is None:
            print("Classification failed")
            return {"status": "error", "message": "Classification failed"}

        if np.isnan(confidence):
            print("Classification confidence is NaN")
            # Check for NaN values in preview images
            for i, img in enumerate(preview_images):
                img_array = np.array(img)
                nan_count = np.isnan(img_array).sum()
                if nan_count > 0:
                    print(f"Preview image {i} contains {nan_count} NaN values")
                else:
                    print(f"Preview image {i} has no NaN values")
            raise ValueError("Classification confidence is NaN")

        # Get Firestore client
        db = firestore.client()

        # Update the document with classification results
        video_ref = db.collection("videos").document(video_id)
        update_data = {
            "class_id": asdict(prediction),
            "confidence": float(confidence),
            "process_date": datetime.now(timezone.utc),
        }

        video_ref.update(update_data)

        print(
            f"Video {video_id} classified as: {prediction} (confidence: {confidence:.3f})"
        )

        return {
            "status": "success",
            "message": f"Video classified as: {prediction}",
            "class_id": str(prediction),
            "confidence": float(confidence),
        }

    except Exception as e:
        print(f"Error processing video classification: {e}")
        return {"status": "error", "message": f"Error processing video: {str(e)}"}
