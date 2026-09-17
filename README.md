# zak-nlptk

A lightweight Python Natural Language Processing toolkit focused on
understanding and implementing core NLP components from first principles.

`zak-nlptk` currently provides tools for:

- Text normalization
- Sentence segmentation
- Word segmentation
- Lemmatization
- Byte Pair Encoding (BPE)
- Token-to-ID encoding
- ID-to-token decoding
- A simple end-to-end NLP pipeline
- A built-in Moby Dick corpus for experimentation

The project is intentionally lightweight and currently relies only on
the Python standard library.

---

## Installation

Install the latest release from PyPI:

```bash
pip install zak-nlptk

You can then import the library:

```python
from zak_nlptk import (
    Corpus,
    TextNormalizer,
    Segmentation,
    BPE,
    Codec,
    Lemmatizer,
    Pipeline
)
```

---

# Quick Start

## 1. Load the built-in corpus

`zak-nlptk` includes a Moby Dick corpus for experimentation.

```python
from zak_nlptk import Corpus

corpus = Corpus()

print(len(corpus.text))
print(corpus.text[:500])
```

The built-in corpus is stored as package data and is loaded automatically
when `Corpus()` is created.

---

## 2. Use your own corpus

You are not limited to the built-in corpus.

You can provide text directly:

```python
from zak_nlptk import Corpus

corpus = Corpus(
    text="This is my own corpus."
)

print(corpus.text)
```

Or load a text file:

```python
from zak_nlptk import Corpus

corpus = Corpus.from_file(
    "my_corpus.txt"
)
```

---

# Text Normalization

The `TextNormalizer` provides configurable text normalization.

You can control:

- case normalization
- Unicode normalization
- whitespace normalization
- diacritic removal

For example:

```python
from zak_nlptk import TextNormalizer

normalizer = TextNormalizer(
    lowercase=True,
    remove_diacritics=True
)

text = "  Héllo   WORLD!  "

normalized = normalizer.normalize(text)

print(normalized)
```

The normalizer does not force lowercase or diacritic removal unless
you request it.

For example:

```python
normalizer = TextNormalizer(
    lowercase=False,
    remove_diacritics=False
)
```

---

# Segmentation

Segmentation is available independently from BPE.

```python
from zak_nlptk import Segmentation

text = """
Hello, world! How are you?
Contact me at john@example.com.
"""

segmentation = Segmentation(text)
```

## Sentence segmentation

```python
sentences = segmentation.sentence_segmentation()

print(sentences)
```

## Word segmentation

```python
words = segmentation.word_segmentation()

print(words)
```

The segmentation component recognizes structures such as:

- email addresses
- URLs and domains
- social handles
- hashtags
- numbers containing commas
- decimal numbers

For example:

```text
john@example.com
https://example.com
@john
#nlp
1,200.50
```

These structures are treated differently from ordinary punctuation.

---

# Byte Pair Encoding

The main tokenizer provided by `zak-nlptk` is a simple implementation
of Byte Pair Encoding (BPE).

The implementation separates BPE training from BPE usage.

## Training a BPE tokenizer

```python
from zak_nlptk import Corpus, BPE

corpus = Corpus()

bpe = BPE()

bpe.train(
    corpus.text,
    vocabulary_size=10_000
)
```

During training, BPE learns:

1. an initial vocabulary
2. frequent adjacent token pairs
3. merge rules
4. a final vocabulary
5. token-to-ID mappings

---

# Tokenization

After training:

```python
tokens = bpe.tokenize(
    "Call me Ishmael."
)

print(tokens)
```

The resulting tokens may contain complete words as well as
subword units learned during BPE training.

For example, a word may be represented as:

```python
["low", "er"]
```

rather than:

```python
["lower"]
```

depending on the learned vocabulary and merge rules.

---

# Encoding Tokens

The BPE tokenizer contains a codec that maps tokens to integer IDs.

```python
ids = bpe.encode(tokens)

print(ids)
```

Conceptually:

```text
["the", "low", "er"]
        ↓
[125, 817, 32]
```

The integer IDs are useful because machine-learning models operate on
numeric representations rather than arbitrary strings.

The mapping is deterministic for a trained tokenizer.

---

# Decoding IDs

The reverse operation is also available:

```python
tokens = bpe.decode(ids)

print(tokens)
```

Conceptually:

```text
[125, 817, 32]
        ↓
["the", "low", "er"]
```

The codec therefore provides two mappings:

```text
token → ID
ID → token
```

---

# Saving a Trained BPE Tokenizer

Training a BPE tokenizer can be expensive for a large corpus.

Once a tokenizer has been trained, save it:

```python
bpe.save(
    "moby_dick_bpe.json"
)
```

The saved model contains the information necessary to reconstruct the
trained tokenizer, including:

- vocabulary
- merge rules
- token-to-ID mappings

---

# Loading a Trained BPE Tokenizer

A previously trained tokenizer can be loaded without retraining:

```python
from zak_nlptk import BPE

bpe = BPE()

bpe.load(
    "moby_dick_bpe.json"
)
```

You can then use it immediately:

```python
tokens = bpe.tokenize(
    "This is new text."
)

ids = bpe.encode(tokens)

print(tokens)
print(ids)
```

This is important because the tokenizer configuration must remain
consistent between training and later use.

---

# BPE Training and New Data

BPE training should normally happen on the training corpus.

After training:

```text
Training corpus
      ↓
BPE training
      ↓
Vocabulary
Merge rules
Token IDs
      ↓
Frozen tokenizer
```

New text should then be processed using the already-trained tokenizer:

```text
New text
   ↓
Frozen BPE tokenizer
   ↓
Tokens
   ↓
Token IDs
```

The tokenizer should not learn new merge rules from test or production
data.

---

# Codec

The `Codec` class can also be used independently.

```python
from zak_nlptk import Codec

codec = Codec()

codec.build({
    "hello",
    "world",
    "!"
})
```

Encode tokens:

```python
ids = codec.encode(
    ["hello", "world", "!"]
)

print(ids)
```

Decode IDs:

```python
tokens = codec.decode(ids)

print(tokens)
```

The codec creates a deterministic mapping between vocabulary items and
integer IDs.

The mapping can also be saved and loaded independently:

```python
codec.save("vocabulary.json")
```

Then:

```python
codec = Codec()

codec.load("vocabulary.json")
```

---

# Lemmatization

`Lemmatizer` is provided as an independent linguistic utility.

It is intentionally separate from the BPE pipeline.

```python
from zak_nlptk import Lemmatizer

lemmatizer = Lemmatizer()

print(
    lemmatizer.lemmatize(
        "running",
        "VERB"
    )
)
```

You can also analyze a word:

```python
result = lemmatizer.analyze(
    "running"
)

print(result)
```

A result contains:

```python
{
    "word": "running",
    "pos": "VERB",
    "lemma": "run"
}
```

The current lemmatizer is a lightweight rule-based implementation and
is intended primarily for experimentation and learning.

It is not intended to replace a full statistical or neural linguistic
parser.

The irregular noun and verb information is stored externally in JSON
files rather than being embedded directly in the Python source.

---

# End-to-End Pipeline

For users who want the main components connected together,
`zak-nlptk` provides `Pipeline`.

```python
from zak_nlptk import Corpus, Pipeline

corpus = Corpus()

pipeline = Pipeline(
    lowercase=True,
    remove_diacritics=True
)

pipeline.train_bpe(
    corpus.text,
    vocabulary_size=10_000
)
```

Then:

```python
ids = pipeline.encode(
    "Call me Ishmael."
)

print(ids)
```

Decode:

```python
tokens = pipeline.decode(ids)

print(tokens)
```

The pipeline performs:

```text
Raw text
   ↓
Normalization
   ↓
Segmentation
   ↓
BPE tokenization
   ↓
Token IDs
```

---

# Architecture

The project is organized into independent components.

```text
                    zak_nlptk
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       Corpus      Normalizer    Lemmatizer
          │             │
          │             ↓
          │        Segmentation
          │             │
          └─────────────┤
                        ↓
                       BPE
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
        Merge Rules          Vocabulary
                                  │
                                  ↓
                                Codec
                                  │
                          ┌───────┴───────┐
                          ↓               ↓
                     Token → ID       ID → Token
```

---

# Project Structure

```text
zak-nlptk/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── zak_nlptk/
│       ├── __init__.py
│       ├── corpus.py
│       ├── normalizer.py
│       ├── segmentation.py
│       ├── lemmatizer.py
│       ├── bpe.py
│       ├── codec.py
│       ├── pipeline.py
│       │
│       └── data/
│           ├── _whale.txt
│           ├── irregular_nouns.json
│           └── irregular_verbs.json
│
└── tests/
    ├── test_bpe.py
    ├── test_codec.py
    └── test_pipeline.py
```

---

# Design Philosophy

`zak-nlptk` is designed as a small educational and experimental NLP
toolkit.

The implementation intentionally exposes the major stages of a basic
NLP processing pipeline instead of hiding everything behind a single
high-level tokenizer.

The major components are:

### Corpus

Provides text data for experimentation.

### TextNormalizer

Performs configurable text normalization.

### Segmentation

Provides sentence and word segmentation.

### BPE

Learns subword units from a corpus and applies learned merge rules.

### Codec

Maps tokens to integer IDs and IDs back to tokens.

### Lemmatizer

Provides lightweight rule-based linguistic analysis.

### Pipeline

Connects normalization, BPE tokenization, and encoding into a simpler
high-level interface.

---

# Current Scope

The project currently focuses on:

- educational NLP implementations
- lightweight experimentation
- understanding tokenizer internals
- BPE training
- vocabulary construction
- token-to-ID mappings
- basic text normalization
- basic segmentation
- lightweight rule-based lemmatization

The project does not currently attempt to provide:

- a production-grade linguistic parser
- a full POS tagger
- contextual lemmatization
- neural language models
- transformer architectures
- statistical sentence boundary detection
- multilingual morphological analysis

Those may be considered in future versions.

---

# Development

Clone the repository:

```bash
git clone <repository-url>
cd zak-nlptk
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the package in editable mode:

```bash
pip install -e .
```

---

# Running Tests

The project uses `pytest` for automated tests.

Run:

```bash
python -m pytest
```

If `pytest` is not installed:

```bash
pip install pytest
```

The test suite is intended to verify the behavior of the major
components, including BPE tokenization, codec encoding/decoding, and
the high-level pipeline.

---

# Building the Package

Install the Python build tool:

```bash
python -m pip install build
```

Build the distribution:

```bash
python -m build
```

This produces distribution files in:

```text
dist/
```

Typically, the directory will contain files similar to:

```text
zak_nlptk-0.1.0-py3-none-any.whl
zak_nlptk-0.1.0.tar.gz
```

The wheel can be installed locally with:

```bash
pip install dist/zak_nlptk-0.1.0-py3-none-any.whl
```

---

# Versioning

The project follows semantic versioning principles.

Versions generally follow:

```text
MAJOR.MINOR.PATCH
```

For example:

```text
0.1.0
```

A patch release generally contains bug fixes.

A minor release can introduce new backwards-compatible functionality.

A major release may contain breaking API changes.

During early development, versions below `1.0.0` indicate that the API
may still change as the project evolves.

---

# License

`zak-nlptk` is distributed under the MIT License.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Author

Created by **Ihimbru Ongum**.

`zak-nlptk` is an independent NLP learning and experimentation project.

---

# Future Development

Possible future improvements include:

- improved English lemmatization
- better irregular-word handling
- improved sentence boundary detection
- configurable tokenization rules
- multilingual normalization
- multilingual segmentation
- improved BPE training efficiency
- deterministic BPE tie-breaking
- tokenizer vocabulary statistics
- tokenizer evaluation utilities
- more comprehensive test coverage
- additional corpus utilities
- support for other subword tokenization algorithms
- improved model serialization
- documentation and usage examples

The project is intentionally kept small so that the underlying NLP
operations remain understandable and inspectable.