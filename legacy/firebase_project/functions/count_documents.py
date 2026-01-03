from fb_help import count_documents, db
from firebase_admin import firestore


def count_docs() -> dict:
    """
    Firestore function that runs once an hour to count documents in the 'videos' and 'pis' collections.
    """
    try:
        # Count documents in the 'videos' collection
        video_count = count_documents("videos")

        # Count documents in the 'pis' collection
        pi_count = count_documents("pis")

        # Optionally, store the counts in a Firestore collection for tracking
        stats_ref = db.collection("stats").document("hourly_counts")
        stats_ref.set(
            {
                "videos_count": video_count,
                "pis_count": pi_count,
                "timestamp": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return {
            "status": "success",
            "message": "Document counts updated successfully.",
            "videos_count": video_count,
            "pis_count": pi_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
