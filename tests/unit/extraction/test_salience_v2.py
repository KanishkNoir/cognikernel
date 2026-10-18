from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cognikernel.extraction import salience_v2


class _Encoding:
    def __init__(self, text: str) -> None:
        self.ids = [ord(char) % 31 + 1 for char in text] or [1]


class _Tokenizer:
    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        return [_Encoding(text) for text in texts]


class _Session:
    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, int]] = []

    def run(self, _outputs, inputs):
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        self.batch_shapes.append(ids.shape)
        pooled = (ids * mask).sum(axis=1).astype("float32")[:, None] / 100
        return [np.repeat(pooled, 384, axis=1)]


def test_classify_scored_batch_batches_nonempty_texts_and_matches_scalar(
    monkeypatch,
) -> None:
    session = _Session()
    weights = np.zeros((385, len(salience_v2.LABELS)), dtype="float32")
    weights[0, 1] = 1.0
    monkeypatch.setattr(salience_v2, "_loaded", True)
    monkeypatch.setattr(salience_v2, "_sess", session)
    monkeypatch.setattr(salience_v2, "_tok", _Tokenizer())
    monkeypatch.setattr(salience_v2, "_W", weights)
    monkeypatch.setattr(salience_v2, "_labels", salience_v2.LABELS)

    texts = ["first", " ", "third", "fifth"]
    batched = salience_v2.classify_scored_batch(texts)

    assert session.batch_shapes == [(3, len("first"))]
    assert batched[1] == ("NOISE", 1.0)
    session.batch_shapes.clear()
    scalar = [salience_v2.classify_scored(text) for text in texts]
    assert batched == scalar
    assert len(session.batch_shapes) == 3
