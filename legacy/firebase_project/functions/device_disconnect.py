from datetime import datetime, timedelta, timezone

from fb_help import db
from firebase_admin import db as rtdb
from firebase_admin import firestore, messaging
from firebase_functions import db_fn, scheduler_fn


def check_health() -> dict:
    """
    Checks each pi's health timestamp in the Realtime Database. If the timestamp is more than 10 minutes old,
    sets the connection_status to 'offline'.
    """
    try:
        # Reference to the root of the devices in RTDB
        devices_ref = rtdb.reference("/devices")
        devices = devices_ref.get()
        if not devices:
            return {"status": "no_devices", "message": "No devices found in RTDB."}

        now = datetime.now(timezone.utc)  # same as pi
        offline_count = 0
        for pi_id, pi_data in devices.items():
            health_ts = pi_data.get("health")
            if health_ts is None:
                continue
            # Convert health_ts to datetime for comparison
            health_datetime = datetime.fromisoformat(health_ts)
            if now - health_datetime > timedelta(minutes=10):
                # Set connection_status to 'offline'
                conn_status_ref = devices_ref.child(f"{pi_id}/connection_status")
                conn_status_ref.set("offline")
                offline_count += 1
        return {
            "status": "success",
            "message": f"Checked {len(devices)} devices. Set {offline_count} offline.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def device_disconnect(
    event: db_fn.Event[db_fn.Change],
) -> None:
    """
    Listens to Realtime Database changes for device connection status.
    When a device disconnects, sends push notifications to associated users.
    """
    # Get the device ID from the event parameters
    pi_id = event.params["pi_id"]

    # Get the new and old connection status values
    before_value = event.data.before
    after_value = event.data.after

    # Only proceed if status changed to "offline"
    if after_value == "offline" and before_value != "offline":
        try:
            # Lookup users associated with this device
            # Assuming you have a Firestore collection mapping devices to users
            pi_doc = db.collection("pis").document(pi_id).get()

            if not pi_doc.exists:
                print(f"No users found for device {pi_id}")
                return

            pi_data = pi_doc.to_dict()
            user_ids = pi_data.get("users", [])

            if not user_ids:
                print(f"No user IDs found for device {pi_id}")
                return

            # Send push notifications to all associated users
            for user_id in user_ids:
                try:
                    # Get user document to retrieve FCM token
                    user_doc = db.collection("users").document(user_id).get()

                    if not user_doc.exists:
                        print(f"User {user_id} not found")
                        continue

                    user_data = user_doc.to_dict()
                    fcm_token = user_data.get("fcm_token")

                    if not fcm_token:
                        print(f"No FCM token found for user {user_id}")
                        continue

                    # Validate FCM token format (basic check)
                    if len(fcm_token) < 50 or not isinstance(fcm_token, str):
                        print(
                            f"Invalid FCM token format for user {user_id}, removing it"
                        )
                        db.collection("users").document(user_id).update(
                            {"fcm_token": firestore.DELETE_FIELD}
                        )
                        continue

                    # Create and send push notification
                    message = messaging.Message(
                        notification=messaging.Notification(
                            title="Device Disconnected",
                            body=f"Your device {pi_id} has disconnected and may need attention.",
                        ),
                        data={
                            "device_id": pi_id,
                            "event_type": "device_disconnected",
                            "timestamp": str(firestore.SERVER_TIMESTAMP),
                        },
                        token=fcm_token,
                    )

                    # Send the message
                    response = messaging.send(message)
                    print(
                        f"Successfully sent notification to user {user_id} for device {pi_id}: {response}"
                    )

                except messaging.UnregisteredError:
                    # FCM token is invalid/expired - remove it from user document
                    print(
                        f"FCM token for user {user_id} is invalid/expired, removing it"
                    )
                    db.collection("users").document(user_id).update(
                        {"fcm_token": firestore.DELETE_FIELD}
                    )
                except messaging.SenderIdMismatchError:
                    # FCM token belongs to a different Firebase project
                    print(
                        f"FCM token for user {user_id} belongs to different project, removing it"
                    )
                    db.collection("users").document(user_id).update(
                        {"fcm_token": firestore.DELETE_FIELD}
                    )
                except messaging.QuotaExceededError:
                    # Rate limit exceeded
                    print(f"FCM quota exceeded for user {user_id}, will retry later")
                except messaging.ThirdPartyAuthError:
                    # APNS/Web Push auth error - usually a project configuration issue
                    print(
                        f"Authentication error with APNS/Web Push for user {user_id}. Check Firebase project configuration."
                    )
                except Exception as notification_error:
                    # Log the specific error type for debugging
                    error_type = type(notification_error).__name__
                    print(
                        f"Failed to send notification to user {user_id} - {error_type}: {str(notification_error)}"
                    )

                    # If it's a general invalid argument error, likely bad token
                    if "invalid-argument" in str(notification_error).lower():
                        print(f"Removing invalid FCM token for user {user_id}")
                        db.collection("users").document(user_id).update(
                            {"fcm_token": firestore.DELETE_FIELD}
                        )

        except Exception as e:
            print(f"Error processing device disconnection for {pi_id}: {str(e)}")
