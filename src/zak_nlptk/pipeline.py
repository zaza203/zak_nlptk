from .bpe import BPE
from .normalizer import TextNormalizer


class Pipeline:

    def __init__(
        self,
        lowercase: bool = False,
        remove_diacritics: bool = True
    ):

        self.normalizer = TextNormalizer(
            lowercase=lowercase,
            remove_diacritics=remove_diacritics
        )

        self.bpe = BPE()

    def normalize(self, text: str) -> str:

        return self.normalizer.normalize(text)

    def train_bpe(
        self,
        text: str,
        vocabulary_size: int
    ):

        normalized_text = self.normalize(
            text
        )

        return self.bpe.train(
            normalized_text,
            vocabulary_size
        )

    def tokenize(self, text: str) -> list[str]:

        normalized_text = self.normalize(
            text
        )

        return self.bpe.tokenize(
            normalized_text
        )

    def encode(self, text: str) -> list[int]:

        tokens = self.tokenize(text)

        return self.bpe.encode(tokens)

    def decode(self, ids: list[int]) -> list[str]:

        return self.bpe.decode(ids)

    def save(self, path: str):

        self.bpe.save(path)

    def load(self, path: str):

        self.bpe.load(path)