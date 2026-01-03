import time

from fb_help import db
from firebase_admin import firestore, messaging
from firebase_functions import firestore_fn


def broadcast(event: firestore_fn.Event[firestore_fn.Change]) -> dict:
    try:
        # Get the broadcast document data
        broadcast_data = event.data.to_dict()

        # Validate required fields
        title = broadcast_data.get("title")
        body = broadcast_data.get("body")

        if not title or not isinstance(title, str):
            return {
                "status": "error",
                "message": "Broadcast must have a valid 'title' string",
            }

        if not body or not isinstance(body, str):
            return {
                "status": "error",
                "message": "Broadcast must have a valid 'body' string",
            }

        # Get all users with FCM tokens
        users_ref = db.collection("users")
        users_query = users_ref.where(
            filter=firestore.FieldFilter("fcm_token", "!=", None)
        )
        users = users_query.get()

        if not users:
            return {"status": "no_users", "message": "No users found with FCM tokens"}

        successful_sends = 0
        failed_sends = 0
        tokens_removed = 0

        # Send notification to each user
        for user_doc in users:
            user_id = user_doc.id
            user_data = user_doc.to_dict()
            fcm_token = user_data.get("fcm_token")

            if not fcm_token:
                continue

            try:
                # Validate FCM token format (basic check)
                if len(fcm_token) < 50 or not isinstance(fcm_token, str):
                    db.collection("users").document(user_id).update(
                        {"fcm_token": firestore.DELETE_FIELD}
                    )
                    tokens_removed += 1
                    continue

                # Create and send push notification
                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                    ),
                    data={
                        "event_type": "broadcast",
                        "timestamp": str(int(time.time())),
                    },
                    token=fcm_token,
                )

                # Send the message
                response = messaging.send(message)
                successful_sends += 1

            except messaging.UnregisteredError:
                # FCM token is invalid/expired - remove it from user document
                db.collection("users").document(user_id).update(
                    {"fcm_token": firestore.DELETE_FIELD}
                )
                tokens_removed += 1
            except messaging.SenderIdMismatchError:
                # FCM token belongs to a different Firebase project
                db.collection("users").document(user_id).update(
                    {"fcm_token": firestore.DELETE_FIELD}
                )
                tokens_removed += 1
            except messaging.QuotaExceededError:
                # Rate limit exceeded
                failed_sends += 1
            except messaging.ThirdPartyAuthError:
                # APNS/Web Push auth error - usually a project configuration issue
                failed_sends += 1
            except Exception as notification_error:
                # If it's a general invalid argument error, likely bad token
                if "invalid-argument" in str(notification_error).lower():
                    db.collection("users").document(user_id).update(
                        {"fcm_token": firestore.DELETE_FIELD}
                    )
                    tokens_removed += 1
                else:
                    failed_sends += 1

        return {
            "status": "success",
            "message": f"Broadcast processed. Successful: {successful_sends}, Failed: {failed_sends}, Tokens removed: {tokens_removed}",
            "successful_sends": successful_sends,
            "failed_sends": failed_sends,
            "tokens_removed": tokens_removed,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error processing broadcast: {str(e)}"}
