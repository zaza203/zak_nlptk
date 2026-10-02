import re
import unicodedata


class TextNormalizer:

    def __init__(
        self,
        lowercase: bool = False,
        remove_diacritics: bool = True
    ):
        self.lowercase = lowercase
        self.remove_diacritics = remove_diacritics

    def casefold(self, text: str) -> str:

        if self.lowercase:
            text = text.casefold()

        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:

        return unicodedata.normalize("NFC", text)

    def normalize_diacritics(self, text: str) -> str:

        if not self.remove_diacritics:
            return text

        text = unicodedata.normalize("NFD", text)

        return "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )

    @staticmethod
    def normalize_whitespace(text: str) -> str:

        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, text: str) -> str:

        text = self.normalize_unicode(text)
        text = self.casefold(text)
        text = self.normalize_diacritics(text)
        text = self.normalize_whitespace(text)

        return text