from pathlib import Path

import pytest

from recycla.process_data.parse import _check_if_file_has_uid


@pytest.mark.parametrize(
    "filename,expected",
    [
        # Valid UID: 20 alnum chars, underscore, digits, .jpg
        ("0h2E8QH6L8SPlGXZDqHK_8.jpg", True),
        ("12345678901234567890_123.png", True),
        ("abcdefghijklmnopqrst_42.jpeg", True),
        # Invalid: no underscore
        ("abcdefghijklmnopqrst42.jpg", False),
        # Invalid: more than one underscore
        ("abcdefghijklmnopqrst_42_1.jpg", False),
        # Invalid: less than 20 chars before underscore
        ("abcdefghijklmnopqr_42.jpg", False),
        # Invalid: more than 20 chars before underscore
        ("abcdefghijklmnopqrstuvwx_42.jpg", False),
        # Invalid: non-alnum in first part
        ("abcdefghij!klmnopqrs_42.jpg", False),
        # Invalid: non-digit after underscore
        ("abcdefghijklmnopqrst_4a2.jpg", False),
        # Invalid: empty before underscore
        ("_42.jpg", False),
        # Invalid: empty after underscore
        ("abcdefghijklmnopqrst_.jpg", False),
        # Valid: digits in first part (still alnum)
        ("12345678901234567890_1.jpg", True),
        # Valid: uppercase letters
        ("ABCDEFGHIJKLMNOQRSTU_123.jpg", True),
    ],
)
def test_check_if_file_has_uid(filename, expected):
    path = Path(filename)
    has_uid, _uid = _check_if_file_has_uid(path)
    assert has_uid == expected
