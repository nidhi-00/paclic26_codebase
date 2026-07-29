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


def test_mlm_prepare_window_with_fast_tokenizer_interface() -> None:
    from src.surprisal_mlm import prepare_window

    class FakeFastTokenizer:
        def build_inputs_with_special_tokens(self, token_ids):
            return [101, *token_ids, 102]

        def get_special_tokens_mask(
            self,
            token_ids,
            already_has_special_tokens=False,
        ):
            assert already_has_special_tokens is True
            return [1] + [0] * (len(token_ids) - 2) + [1]

    input_ids, positions = prepare_window(
        FakeFastTokenizer(),
        [10, 11, 12, 13],
        1,
        3,
    )

    assert input_ids.tolist() == [101, 11, 12, 102]
    assert positions == [1, 2]
