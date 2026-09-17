from pathlib import Path

import pytest

from zak_nlptk import BPE


def test_bpe_training_creates_vocabulary():
    bpe = BPE()

    vocabulary = bpe.train(
        "low lower lowest",
        vocabulary_size=20,
    )

    assert vocabulary
    assert bpe.vocabulary
    assert bpe.merge_rules
    assert bpe.codec.token_to_id


def test_bpe_tokenizes_after_training():
    bpe = BPE()

    bpe.train(
        "low lower lowest",
        vocabulary_size=20,
    )

    tokens = bpe.tokenize("lower")

    assert tokens
    assert all(isinstance(token, str) for token in tokens)


def test_bpe_preserves_email():
    bpe = BPE()

    bpe.train(
        "contact test@example.com today",
        vocabulary_size=20,
    )

    tokens = bpe.tokenize("test@example.com")

    assert tokens == ["test@example.com"]


def test_bpe_preserves_url():
    bpe = BPE()

    bpe.train(
        "visit https://example.com today",
        vocabulary_size=20,
    )

    tokens = bpe.tokenize("https://example.com")

    assert tokens == ["https://example.com"]


def test_bpe_preserves_hashtag():
    bpe = BPE()

    bpe.train(
        "#python is useful",
        vocabulary_size=20,
    )

    tokens = bpe.tokenize("#python")

    assert tokens == ["#python"]


def test_bpe_preserves_number():
    bpe = BPE()

    bpe.train(
        "The price is 6.7 dollars",
        vocabulary_size=20,
    )

    tokens = bpe.tokenize("6.7")

    assert tokens == ["6.7"]


def test_bpe_encode_and_decode():
    bpe = BPE()

    bpe.train(
        "low lower lowest",
        vocabulary_size=20,
    )

    tokens = bpe.tokenize("lower")
    ids = bpe.encode(tokens)
    decoded = bpe.decode(ids)

    assert decoded == tokens


def test_bpe_save_and_load(tmp_path: Path):
    bpe = BPE()

    bpe.train(
        "low lower lowest",
        vocabulary_size=20,
    )

    original_tokens = bpe.tokenize("lower")
    original_ids = bpe.encode(original_tokens)

    path = tmp_path / "bpe.json"

    bpe.save(path)

    loaded = BPE()
    loaded.load(path)

    assert loaded.vocabulary == bpe.vocabulary
    assert loaded.merge_rules == bpe.merge_rules
    assert loaded.codec.token_to_id == bpe.codec.token_to_id

    loaded_tokens = loaded.tokenize("lower")
    loaded_ids = loaded.encode(loaded_tokens)

    assert loaded_tokens == original_tokens
    assert loaded_ids == original_ids


def test_bpe_requires_training_before_tokenization():
    bpe = BPE()

    with pytest.raises(ValueError, match="not been trained or loaded"):
        bpe.tokenize("hello")


def test_bpe_rejects_invalid_vocabulary_size():
    bpe = BPE()

    with pytest.raises(ValueError, match="greater than 0"):
        bpe.train("hello world", vocabulary_size=0)