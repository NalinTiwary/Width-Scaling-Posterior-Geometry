import pytest

from bnn_geometry.paper_export import _retained_segments, h_dir_name
from bnn_geometry.storage import StorageError


def test_h_dir_name_matches_campaign_layout():
    assert h_dir_name(0.01) == "h_0p01"
    assert h_dir_name(0.005) == "h_0p005"


def test_retained_segments_order():
    assert _retained_segments(["discard", "stage_1", "stage_2", "stage_3"]) == ["stage_1", "stage_2", "stage_3"]
    for bad in (["stage_1", "discard"], ["discard", "stage_2"], ["discard", "stage_1", "measure"],
                ["discard", "stage_1", "stage_3"]):
        with pytest.raises(StorageError):
            _retained_segments(bad)
