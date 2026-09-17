from pathlib import Path

import pytest

from zak_nlptk import Codec


def test_build_creates_token_ids():
    codec = Codec()
    codec.build({"cat", "dog", "the"})

    assert codec.token_to_id == {
        "cat": 0,
        "dog": 1,
        "the": 2,
    }


def test_encode():
    codec = Codec()
    codec.build({"cat", "dog", "the"})

    ids = codec.encode(["the", "cat", "dog"])

    assert ids == [2, 0, 1]


def test_decode():
    codec = Codec()
    codec.build({"cat", "dog", "the"})

    tokens = codec.decode([2, 0, 1])

    assert tokens == ["the", "cat", "dog"]


def test_encode_decode_round_trip():
    codec = Codec()
    codec.build({"cat", "dog", "the"})

    tokens = ["the", "cat", "dog"]

    assert codec.decode(codec.encode(tokens)) == tokens


def test_encode_unknown_token_raises_error():
    codec = Codec()
    codec.build({"cat", "dog"})

    with pytest.raises(ValueError, match="Unknown token"):
        codec.encode(["bird"])


def test_decode_unknown_id_raises_error():
    codec = Codec()
    codec.build({"cat", "dog"})

    with pytest.raises(ValueError, match="Unknown token ID"):
        codec.decode([99])


def test_save_and_load(tmp_path: Path):
    codec = Codec()
    codec.build({"cat", "dog", "the"})

    path = tmp_path / "codec.json"

    codec.save(path)

    loaded = Codec()
    loaded.load(path)

    assert loaded.token_to_id == codec.token_to_id
    assert loaded.id_to_token == codec.id_to_token
    assert loaded.encode(["the", "cat"]) == [2, 0]