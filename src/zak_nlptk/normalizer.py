# import re
# import unicodedata

# class TextNormalizer:

#     def __init__(self, lowercase: bool = False, remove_diacritics: bool = True):
#         """
#         Initialize the TextNormalizer with optional lowercase normalization.
        
#         Args:
#             lowercase (bool): If True, convert text to lowercase during normalization.
#         """
#         self.lowercase = lowercase
#         self.remove_diacritics = remove_diacritics

#     def casefold(self, text: str) -> str:
#         """
#         Convert the input text to lowercase for case normalization.
        
#         Args:
#             text (str): The input text to be normalized.
        
#         Returns:
#             str: The normalized text in lowercase.
#         """
#         if self.lowercase:
#             text = text.lower()
#         return text

#     def normalize_whitespace(self, text: str) -> str:
#         """
#         Replace multiple consecutive whitespace characters with a single space.
        
#         Args:
#             text (str): The input text to be normalized.
        
#         Returns:
#             str: The normalized text with unnecessary whitespace removed.
#         """
#         return re.sub(r'\s+', " ", text).strip()

#     def normalize_unicode(self, text: str) -> str:
#         """
#         Normalize the input text to Unicode NFC form.
        
#         Args:
#             text (str): The input text to be normalized.
        
#         Returns:
#             str: The normalized text in Unicode NFC form.
#         """
#         return unicodedata.normalize('NFC', text)

#     def normalize_diacritics(self, text: str) -> str:
#         """
#         Remove diacritical marks from the input text.
        
#         Args:
#             text (str): The input text to be normalized.
        
#         Returns:
#             str: The normalized text with diacritical marks removed.
#         """
#         if self.remove_diacritics:
#             text = unicodedata.normalize('NFD', text)
#             text = ''.join(c for c in text if not unicodedata.combining(c))
#         return text

#     def normalize(self, text: str) -> str:
#         """
#         Apply all normalization steps to the input text.
        
#         Args:
#             text (str): The input text to be normalized.
        
#         Returns:
#             str: The fully normalized text.
#         """
#         text = self.casefold(text)
#         text = self.normalize_whitespace(text)
#         text = self.normalize_unicode(text)
#         text = self.normalize_diacritics(text)
#         return text



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