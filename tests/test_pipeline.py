from pathlib import Path

import pytest

from zak_nlptk import Pipeline


def test_pipeline_normalizes_text():
    pipeline = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    result = pipeline.normalize("  HéLLo   Wörld  ")

    assert result == "hello world"


def test_pipeline_can_train_bpe():
    pipeline = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    vocabulary = pipeline.train_bpe(
        "Low lower lowest",
        vocabulary_size=20,
    )

    assert vocabulary
    assert pipeline.bpe.vocabulary


def test_pipeline_tokenize():
    pipeline = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    pipeline.train_bpe(
        "Low lower lowest",
        vocabulary_size=20,
    )

    tokens = pipeline.tokenize("LOWER")

    assert tokens
    assert all(isinstance(token, str) for token in tokens)


def test_pipeline_encode_decode():
    pipeline = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    pipeline.train_bpe(
        "Low lower lowest",
        vocabulary_size=20,
    )

    tokens = pipeline.tokenize("LOWER")
    ids = pipeline.encode("LOWER")
    decoded = pipeline.decode(ids)

    assert decoded == tokens


def test_pipeline_save_and_load(tmp_path: Path):
    pipeline = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    pipeline.train_bpe(
        "Low lower lowest",
        vocabulary_size=20,
    )

    original_tokens = pipeline.tokenize("LOWER")
    original_ids = pipeline.encode("LOWER")

    path = tmp_path / "pipeline.json"

    pipeline.save(path)

    loaded = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    loaded.load(path)

    assert loaded.tokenize("LOWER") == original_tokens
    assert loaded.encode("LOWER") == original_ids


def test_pipeline_preserves_special_tokens():
    pipeline = Pipeline(
        lowercase=True,
        remove_diacritics=True,
    )

    pipeline.train_bpe(
        "contact test@example.com at https://example.com",
        vocabulary_size=30,
    )

    tokens = pipeline.tokenize(
        "CONTACT TEST@EXAMPLE.COM at https://example.com"
    )

    assert "test@example.com" in tokens
    assert "https://example.com" in tokens