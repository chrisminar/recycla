from recycla.process_data.io import import_classnames


def test_import_classnames():
    primary_classnames, secondary_classnames = import_classnames()

    assert isinstance(primary_classnames, list)
    assert isinstance(secondary_classnames, list)

    # Check if the primary classnames are sorted
    assert list(primary_classnames) == sorted(primary_classnames)
    # Check if the secondary classnames are sorted
    assert list(secondary_classnames) == sorted(secondary_classnames)

    # Check if the primary classnames are unique
    assert len(primary_classnames) == len(set(primary_classnames))
    # Check if the secondary classnames are unique
    assert len(secondary_classnames) == len(set(secondary_classnames))

    assert len(primary_classnames) > 0
    assert len(secondary_classnames) > 0
