from __future__ import annotations

from src.scoring import target_centred_window


def test_target_centred_window_retains_target_at_edges() -> None:
    left = target_centred_window(100, 0, 2, 10)
    assert left.start == 0
    assert left.end == 10
    right = target_centred_window(100, 98, 100, 10)
    assert right.start == 90
    assert right.end == 100
    middle = target_centred_window(100, 49, 51, 10)
    assert middle.start <= 49 < 51 <= middle.end
    assert middle.end - middle.start == 10
