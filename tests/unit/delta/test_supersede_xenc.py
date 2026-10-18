from __future__ import annotations

import numpy as np

from cognikernel.delta import supersede_xenc


class _Encoding:
    def __init__(self, first: str, second: str) -> None:
        self.ids = list(range(1, len(first) + len(second) + 1))
        self.type_ids = [0] * (len(first) + 1) + [1] * (len(second) - 1)


class _Tokenizer:
    def encode_batch(self, pairs: list[tuple[str, str]]) -> list[_Encoding]:
        return [_Encoding(*pair) for pair in pairs]


class _Session:
    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, int]] = []

    def run(self, _outputs, inputs):
        ids = inputs["input_ids"]
        self.batch_shapes.append(ids.shape)
        positive = ids.sum(axis=1).astype("float32")
        return [np.column_stack([np.zeros_like(positive), positive])]


def test_prob_supersedes_batches_only_unpadded_equal_length_pairs(monkeypatch) -> None:
    session = _Session()
    monkeypatch.setattr(supersede_xenc, "_loaded", True)
    monkeypatch.setattr(supersede_xenc, "_sess", session)
    monkeypatch.setattr(supersede_xenc, "_tok", _Tokenizer())
    monkeypatch.setattr(supersede_xenc, "_load", lambda: True)

    pairs = [("a", "bb"), ("ab", "c"), ("a", "b"), ("", "ignored")]
    scores = supersede_xenc.prob_supersedes_batch(pairs)

    assert session.batch_shapes == [(2, 3), (1, 2)]
    assert all(score is not None for score in scores[:3])
    assert scores[3] is None
    assert supersede_xenc.prob_supersedes(*pairs[0]) == scores[0]
