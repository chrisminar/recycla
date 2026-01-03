# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import db_fn, firestore_fn, options, scheduler_fn

from broadcast import broadcast
from count_documents import count_docs
from device_disconnect import check_health, device_disconnect
from fb_help import db, get_materials, get_videos, is_interesting_to_read
from pi_summary import update_pi_summary
from reading import image_to_item


@firestore_fn.on_document_updated(
    document="videos/{id}", memory=options.MemoryOption.MB_512
)
def on_video_update(event: firestore_fn.Event[firestore_fn.Change]) -> dict:
    # Get the updated document
    updated_value = event.data.after.to_dict()
    old_value = event.data.before.to_dict()

    user_ground_truth_updated = "user_ground_truth" in updated_value and updated_value[
        "user_ground_truth"
    ] != old_value.get("user_ground_truth")
    class_id_updated = "class_id" in updated_value and updated_value[
        "class_id"
    ] != old_value.get("class_id")
    user_ground_truth_was_null = (
        user_ground_truth_updated and old_value.get("user_ground_truth") is None
    )

    status = ""
    message = ""
    # user ground truth updated
    if user_ground_truth_updated or class_id_updated:
        vids = get_videos(updated_value["piid"])

        update_pi_summary(vids, updated_value, updated_value["piid"])

        status = "video stats updated"
        message = "User document update processed."
    else:
        status = "video stats skipped"
        message = "No relevant fields updated."

    if user_ground_truth_was_null:
        _material, sub_material = get_materials(updated_value)
        if not is_interesting_to_read(sub_material):
            status += " Sub-material not interesting to read, skipping grocery item processing."
            message += (
                " Grocery item processing skipped due to uninteresting sub-material."
            )
        else:
            # If user ground truth was null, process the image to item
            gcs_image_path = updated_value.get("preview_gcs_path")
            if gcs_image_path:
                item = image_to_item(gcs_image_path)
                doc_path = event.document
                db.document(doc_path).update({"grocery_item": item})
                status += " Grocery item processed."
                message += f" Grocery item: {item if item else 'None'}"
            else:
                status += " No image path provided."
                message += " No image path to process grocery item."
    else:
        status += " No grocery item processed."
        message += " Grocery item processing skipped."
    return {"status": status, "message": message}


@firestore_fn.on_document_created(
    document="videos/{id}", memory=options.MemoryOption.GB_4
)
def on_video_create(event: firestore_fn.Event[firestore_fn.Change]) -> dict:
    # Get the updated document
    from inference import run_inference

    return run_inference(event)


@scheduler_fn.on_schedule(
    schedule="every 60 minutes", memory=options.MemoryOption.MB_512
)
def count_documents_hourly(_event: scheduler_fn.ScheduledEvent) -> dict:
    """
    Firestore function that runs once an hour to count documents in the 'videos' and 'pis' collections.
    """
    return count_docs()


@scheduler_fn.on_schedule(
    schedule="every 30 minutes", memory=options.MemoryOption.MB_512
)
def check_pi_health_and_update_status(_event: scheduler_fn.ScheduledEvent) -> dict:
    """
    Checks each pi's health timestamp in the Realtime Database. If the timestamp is more than 10 minutes old,
    sets the connection_status to 'offline'.
    """
    return check_health()


@db_fn.on_value_written(
    reference="/devices/{pi_id}/connection_status", memory=options.MemoryOption.MB_512
)
def on_device_connection_status_change(
    event: db_fn.Event[db_fn.Change],
) -> None:
    """
    Listens to Realtime Database changes for device connection status.
    When a device disconnects, sends push notifications to associated users.
    """
    device_disconnect(event)


@firestore_fn.on_document_created(
    document="broadcasts/{id}", memory=options.MemoryOption.MB_512
)
def on_broadcast_created_send_push(
    event: firestore_fn.Event[firestore_fn.Change],
) -> dict:
    """
    Triggers when a new document is added to the broadcasts collection.
    Sends push notifications to all users with FCM tokens.
    """
    return broadcast(event)
