# import re


# class Segmentation:

#     def __init__(self, text):
#         self.text = text

#     rules = [
#         r"\w+(?:\.\w+)*@\w+(?:\.\w+)+",     # Email
#         r"(?:https?://|www\.)?[\w-]+(?:\.[\w-]+)+(?:/[^\s]*)?",     # URL
#         r"[@#][\w.-]+",     # Social handles / hashtags
#         r"\d[\d,]*(?:\.\d+)?"       # Number with internal comma/decimal punctuation
#     ]

#     def sentence_segmentation(self):
#         sentences = re.split(r"(?<=[.!?])\s+", self.text.strip())
#         return [sentence.strip() for sentence in sentences if sentence.strip()]

#     def word_segmentation(self) -> list[str]:
#         text = self.text
#         tokens = []
#         position = 0

#         combined_pattern = re.compile(
#             "|".join(f"({rule})" for rule in self.rules)
#         )

#         for match in combined_pattern.finditer(text):

#             before = text[position:match.start()]
#             tokens.extend(self._split_normal_text(before))

#             token = match.group(0)

#             tokens.append(token)

#             position = match.end()

#         remaining = text[position:]
#         tokens.extend(self._split_normal_text(remaining))

#         return tokens

#     @staticmethod
#     def _split_normal_text(text):
#         tokens = []

#         for token in re.findall(r"\w+|[^\w\s]", text):
#             tokens.append(token)

#         return tokens



import re


class Segmentation:

    rules = [
        # Email
        r"\w+(?:\.\w+)*@\w+(?:\.\w+)+",

        # URL / domain
        r"(?:https?://|www\.)?[\w-]+(?:\.[\w-]+)+(?:/[^\s]*)?",

        # Social handles / hashtags
        r"[@#][\w.-]+",

        # Numbers with internal commas / decimals
        r"\d[\d,]*(?:\.\d+)?"
    ]

    def __init__(self, text: str):

        self.text = text

    def sentence_segmentation(self) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            self.text.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def word_segmentation(self) -> list[str]:

        tokens = []

        combined_pattern = re.compile(
            "|".join(
                f"({rule})"
                for rule in self.rules
            )
        )

        position = 0

        for match in combined_pattern.finditer(self.text):

            before = self.text[position:match.start()]

            tokens.extend(
                self._split_normal_text(before)
            )

            tokens.append(match.group())

            position = match.end()

        remaining = self.text[position:]

        tokens.extend(
            self._split_normal_text(remaining)
        )

        return tokens

    @staticmethod
    def _split_normal_text(text: str) -> list[str]:

        return re.findall(
            r"\w+|[^\w\s]",
            text
        )