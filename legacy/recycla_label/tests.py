def test_bad_path():
    with pytest.raises(FileNotFoundError):
        get_videos(start_path / "non_existent_path")


def test_get_videos():
    videos = get_videos(start_path)
    assert len(videos) > 0, "No videos found in the specified path."
    for video in videos:
        assert video.suffix == ".h264", f"Found non-h264 file: {video}"


def test_sub_material_to_material():
    primary_class, second_class = import_classnames()
    for sub_material in second_class:
        material = sub_material_to_material(sub_material)
        assert (
            material in primary_class
        ), f"Sub-material {sub_material} does not map to a primary class."
