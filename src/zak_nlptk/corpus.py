from pathlib import Path


class Corpus:

    def __init__(self, text: str | None = None):

        if text is not None:
            self.text = text
        else:
            self.text = self._load_whale()

    @staticmethod
    def _load_whale() -> str:

        path = Path(__file__).parent / "data" / "the_whale.txt"

        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    @classmethod
    def from_file(cls, path: str) -> "Corpus":

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        return cls(text=text)