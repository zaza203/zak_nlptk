# from collections import defaultdict
# import json
# import re


# class BPE:

#     rules = [
#         r"\w+(?:\.\w+)*@\w+(?:\.\w+)+",
#         r"(?:https?://|www\.)?[\w-]+(?:\.[\w-]+)+(?:/[^\s]*)?",
#         r"[@#][\w.-]+",
#         r"\d[\d,]*(?:\.\d+)?"
#     ]

#     def __init__(self, text: list[str] | None = None):
#         self.text = text or []
#         self.merge_rules = []
#         self.vocabulary = set()

#     def _pre_tokenize(self, word: str) -> list[str]:

#         pattern = re.compile("|".join(
#             f"({rule})" for rule in self.rules
#         ))

#         tokens = []
#         position = 0

#         for match in pattern.finditer(word):

#             before = word[position:match.start()]

#             tokens.extend(
#                 re.findall(r"\w+|[^\w\s]", before)
#             )

#             tokens.append(match.group())

#             position = match.end()

#         remaining = word[position:]

#         tokens.extend(
#             re.findall(r"\w+|[^\w\s]", remaining)
#         )

#         return tokens

#     def _prepare_training_data(self) -> list[list[str]]:

#         sequences = []

#         for word in self.text:

#             pre_tokens = self._pre_tokenize(word)

#             for token in pre_tokens:

#                 # Special structures remain a single token.
#                 if any(
#                     re.fullmatch(rule, token)
#                     for rule in self.rules
#                 ):
#                     sequences.append([token])

#                 else:
#                     sequences.append(list(token))

#         return sequences

#     @staticmethod
#     def _count_pairs(sequences: list[list[str]]) -> dict[tuple[str, str], int]:

#         pair_counts = defaultdict(int)

#         for tokens in sequences:

#             for i in range(len(tokens) - 1):

#                 pair = (tokens[i], tokens[i + 1])

#                 pair_counts[pair] += 1

#         return pair_counts

#     @staticmethod
#     def _merge_pair(sequences: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:

#         left, right = pair
#         merged_token = left + right

#         new_sequences = []

#         for tokens in sequences:

#             new_tokens = []
#             i = 0

#             while i < len(tokens):

#                 if (
#                     i < len(tokens) - 1
#                     and tokens[i] == left
#                     and tokens[i + 1] == right
#                 ):
#                     new_tokens.append(merged_token)
#                     i += 2

#                 else:
#                     new_tokens.append(tokens[i])
#                     i += 1

#             new_sequences.append(new_tokens)

#         return new_sequences

#     def train(self, vocabulary_size: int):

#         sequences = self._prepare_training_data()

#         self.vocabulary = {
#             token
#             for sequence in sequences
#             for token in sequence
#         }

#         while len(self.vocabulary) < vocabulary_size:

#             pair_counts = self._count_pairs(sequences)

#             if not pair_counts:
#                 break

#             best_pair = max(
#                 pair_counts,
#                 key=pair_counts.get
#             )

#             left, right = best_pair
#             new_token = left + right

#             self.merge_rules.append(best_pair)

#             sequences = self._merge_pair(
#                 sequences,
#                 best_pair
#             )

#             self.vocabulary.add(new_token)

#         return self.vocabulary

#     def tokenize(self, text: str) -> list[str]:

#         if not self.merge_rules:
#             raise ValueError(
#                 "BPE has not been trained."
#             )

#         tokens = []

#         for word in text.split():

#             pre_tokens = self._pre_tokenize(word)

#             for token in pre_tokens:

#                 if any(
#                     re.fullmatch(rule, token)
#                     for rule in self.rules
#                 ):
#                     tokens.append(token)
#                     continue

#                 current = list(token)

#                 for pair in self.merge_rules:

#                     current = self._merge_token(
#                         current,
#                         pair
#                     )

#                 tokens.extend(current)

#         return tokens

#     @staticmethod
#     def _merge_token(
#         tokens: list[str],
#         pair: tuple[str, str]
#     ) -> list[str]:

#         left, right = pair
#         result = []
#         i = 0

#         while i < len(tokens):

#             if (
#                 i < len(tokens) - 1
#                 and tokens[i] == left
#                 and tokens[i + 1] == right
#             ):
#                 result.append(left + right)
#                 i += 2

#             else:
#                 result.append(tokens[i])
#                 i += 1

#         return result

#     def save(self, path: str):

#         data = {
#             "vocabulary": sorted(self.vocabulary),
#             "merge_rules": [
#                 list(pair)
#                 for pair in self.merge_rules
#             ]
#         }

#         with open(path, "w", encoding="utf-8") as file:
#             json.dump(
#                 data,
#                 file,
#                 ensure_ascii=False,
#                 indent=2
#             )

#     def load(self, path: str):

#         with open(path, "r", encoding="utf-8") as file:
#             data = json.load(file)

#         self.vocabulary = set(
#             data["vocabulary"]
#         )

#         self.merge_rules = [
#             tuple(pair)
#             for pair in data["merge_rules"]
#         ]



from collections import defaultdict
import json
import re

from .codec import Codec
from .segmentation import Segmentation


class BPE:

    rules = Segmentation.rules

    def __init__(self):

        self.merge_rules = []
        self.vocabulary = set()
        self.codec = Codec()

    @staticmethod
    def _is_special_token(token: str) -> bool:

        return any(
            re.fullmatch(rule, token)
            for rule in BPE.rules
        )

    def _prepare_training_data(
        self,
        text: str
    ) -> list[list[str]]:

        segmentation = Segmentation(text)

        words = segmentation.word_segmentation()

        sequences = []

        for word in words:

            if self._is_special_token(word):

                sequences.append([word])

            else:

                sequences.append(
                    list(word)
                )

        return sequences

    @staticmethod
    def _count_pairs(
        sequences: list[list[str]]
    ) -> dict[tuple[str, str], int]:

        pair_counts = defaultdict(int)

        for tokens in sequences:

            for i in range(len(tokens) - 1):

                pair = (
                    tokens[i],
                    tokens[i + 1]
                )

                pair_counts[pair] += 1

        return pair_counts

    @staticmethod
    def _merge_pair(
        sequences: list[list[str]],
        pair: tuple[str, str]
    ) -> list[list[str]]:

        left, right = pair

        merged_token = left + right

        new_sequences = []

        for tokens in sequences:

            new_tokens = []

            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):

                    new_tokens.append(
                        merged_token
                    )

                    i += 2

                else:

                    new_tokens.append(
                        tokens[i]
                    )

                    i += 1

            new_sequences.append(
                new_tokens
            )

        return new_sequences

    def train(
        self,
        text: str,
        vocabulary_size: int
    ) -> set[str]:

        if vocabulary_size <= 0:
            raise ValueError(
                "vocabulary_size must be greater than 0."
            )

        self.merge_rules = []

        sequences = self._prepare_training_data(text)

        self.vocabulary = {
            token
            for sequence in sequences
            for token in sequence
        }

        while len(self.vocabulary) < vocabulary_size:

            pair_counts = self._count_pairs(
                sequences
            )

            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=pair_counts.get
            )

            left, right = best_pair

            new_token = left + right

            if new_token in self.vocabulary:
                break

            self.merge_rules.append(
                best_pair
            )

            sequences = self._merge_pair(
                sequences,
                best_pair
            )

            self.vocabulary.add(
                new_token
            )

        self.codec.build(
            self.vocabulary
        )

        return self.vocabulary

    def _tokenize_word(
        self,
        token: str
    ) -> list[str]:

        if self._is_special_token(token):

            return [token]

        current = list(token)

        for pair in self.merge_rules:

            current = self._merge_token(
                current,
                pair
            )

        return current

    def tokenize(self, text: str) -> list[str]:

        if not self.merge_rules:

            raise ValueError(
                "BPE has not been trained or loaded."
            )

        segmentation = Segmentation(text)

        words = segmentation.word_segmentation()

        tokens = []

        for word in words:

            tokens.extend(
                self._tokenize_word(word)
            )

        return tokens

    @staticmethod
    def _merge_token(
        tokens: list[str],
        pair: tuple[str, str]
    ) -> list[str]:

        left, right = pair

        result = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == left
                and tokens[i + 1] == right
            ):

                result.append(
                    left + right
                )

                i += 2

            else:

                result.append(
                    tokens[i]
                )

                i += 1

        return result

    def encode(
        self,
        tokens: list[str]
    ) -> list[int]:

        return self.codec.encode(tokens)

    def decode(
        self,
        ids: list[int]
    ) -> list[str]:

        return self.codec.decode(ids)

    def save(self, path: str):

        data = {
            "vocabulary": sorted(
                self.vocabulary
            ),

            "merge_rules": [
                list(pair)
                for pair in self.merge_rules
            ],

            "token_to_id": self.codec.token_to_id
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path: str):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.vocabulary = set(
            data["vocabulary"]
        )

        self.merge_rules = [
            tuple(pair)
            for pair in data["merge_rules"]
        ]

        self.codec.token_to_id = {
            token: int(index)
            for token, index
            in data["token_to_id"].items()
        }

        self.codec.id_to_token = {
            index: token
            for token, index
            in self.codec.token_to_id.items()
        }