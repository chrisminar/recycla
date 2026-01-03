from collections import defaultdict

from fb_help import (
    date_2_month_string,
    date_2_week_string,
    db,
    get_materials,
    get_week_number,
    is_primary_prediction_correct,
    is_secondary_prediction_correct,
    video_is_bounty_valid,
)


def create_pi_summary(vids) -> dict:
    counter = 0
    bounty = 0
    secondary_counter = 0
    primary_correct_counter = 0
    secondary_correct_counter = 0
    primary_cls_correct_counter = defaultdict(int)
    secondary_cls_correct_counter = defaultdict(int)
    primary_cls_counter = defaultdict(int)
    secondary_cls_counter = defaultdict(int)
    primary_cls_gt_only_counter = defaultdict(int)
    secondary_cls_gt_only_counter = defaultdict(int)

    for vid in vids:
        vid_gt = vid.get("user_ground_truth")
        vid_cls = vid.get("class_id")
        primary_correct = is_primary_prediction_correct(vid_gt, vid_cls)
        secondary_correct = is_secondary_prediction_correct(vid_gt, vid_cls)
        material, sub_material = get_materials(vid)
        if material is not None:
            if (
                primary_correct is not None
            ):  # if there is at least one source of material and it wasn't supposed to be skipped
                counter += 1
            if primary_correct:
                primary_correct_counter += 1
                primary_cls_correct_counter[material] += 1
                primary_cls_counter[material] += 1

        if sub_material is not None and secondary_correct is not None:
            bounty += video_is_bounty_valid(vid)
            secondary_counter += 1
            secondary_correct_counter += secondary_correct
            secondary_cls_correct_counter[sub_material] += secondary_correct
            secondary_cls_counter[sub_material] += 1

    primary_correct_pct = (
        primary_correct_counter / (counter) * 100 if counter > 0 else 0
    )
    secondary_correct_counter_pct = (
        secondary_correct_counter / (secondary_counter) * 100
        if secondary_counter > 0
        else 0
    )

    primary_cls_correct_pct = {
        k: primary_cls_correct_counter[k] / primary_cls_counter[k] * 100
        for k in primary_cls_counter
    }
    secondary_cls_correct_pct = {
        k: secondary_cls_correct_counter[k] / secondary_cls_counter[k] * 100
        for k in secondary_cls_counter
    }

    for cls, count in primary_cls_counter.items():
        primary_cls_gt_only_counter[cls] += count
    for cls, count in secondary_cls_counter.items():
        secondary_cls_gt_only_counter[cls] += count

    update_dict = {
        "num_items": counter,
        "bounty": bounty,
        "material_pct_correct": primary_correct_pct,
        "sub_material_pct_correct": secondary_correct_counter_pct,
        "material_num_class": dict(primary_cls_gt_only_counter),
        "sub_material_num_class": dict(secondary_cls_gt_only_counter),
        "material_pct_class_correct": dict(primary_cls_correct_pct),
        "sub_material_pct_class_correct": dict(secondary_cls_correct_pct),
    }

    return update_dict


def get_time_intervals(video_doc_dict: dict) -> tuple[int, str]:
    """Get week and month of video."""
    video_date = video_doc_dict.get("upload_date")
    if video_date is None:
        return None, None
    video_week = get_week_number(video_date)
    video_month = video_date.month
    return video_week, video_month


def time_split_videos(vids, video_doc_dict: dict) -> tuple[list[dict], list[dict]]:
    video_date = video_doc_dict.get("upload_date")
    if video_date is None:
        return None, None
    week, month = get_time_intervals(video_doc_dict)
    weekly_videos = [
        vid
        for vid in vids
        if (get_week_number(vid.get("upload_date")) == week)
        and (vid.get("upload_date") is not None)
    ]
    monthly_videos = [
        vid
        for vid in vids
        if (vid.get("upload_date").month == month)
        and (vid.get("upload_date") is not None)
    ]
    return weekly_videos, monthly_videos


def update_pi_summary(vids, video_doc_dict, pi_id):
    """Update the PI summary document with the latest video data."""
    if vids is None:
        return
    weekly_videos, monthly_videos = time_split_videos(vids, video_doc_dict)

    overall_update_dict = create_pi_summary(vids)

    monthly_doc_id = date_2_month_string(video_doc_dict.get("upload_date"))

    all_ref = db.collection("pis").document(pi_id)
    all_ref.update(overall_update_dict)

    weekly_doc_id = date_2_week_string(video_doc_dict.get("upload_date"))
    if weekly_doc_id is not None and weekly_videos is not None:
        weekly_update_dict = create_pi_summary(weekly_videos)
        week_ref = all_ref.collection("weekly_stats").document(weekly_doc_id)
        week_ref.set(weekly_update_dict)

    if monthly_doc_id is not None and monthly_videos is not None:
        monthly_update_dict = create_pi_summary(monthly_videos)
        month_ref = all_ref.collection("monthly_stats").document(monthly_doc_id)
        month_ref.set(monthly_update_dict)
