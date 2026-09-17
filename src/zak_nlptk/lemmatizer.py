from pathlib import Path
import json
import re


class Lemmatizer:

    PRONOUNS = {
        "i", "me", "my", "mine",
        "you", "your", "yours",
        "he", "him", "his",
        "she", "her", "hers",
        "it", "its",
        "we", "us", "our", "ours",
        "they", "them", "their", "theirs"
    }

    DETERMINERS = {
        "a", "an", "the",
        "this", "that", "these", "those",
        "some", "any", "each", "every",
        "either", "neither", "both", "all", "no"
    }

    PREPOSITIONS = {
        "about", "above", "across", "after", "against",
        "along", "among", "around", "at", "before",
        "behind", "below", "beneath", "beside", "between",
        "beyond", "by", "despite", "during", "except",
        "for", "from", "in", "inside", "into", "near",
        "of", "off", "on", "onto", "over", "through",
        "to", "toward", "under", "until", "up", "upon",
        "with", "within", "without"
    }

    CONJUNCTIONS = {
        "and", "or", "but", "nor", "so", "yet",
        "although", "because", "if", "while"
    }

    AUXILIARIES = {
        "will", "would", "shall", "should",
        "may", "might", "can", "could", "must"
    }

    ADVERB_SUFFIXES = (
        "ly",
        "ward",
        "wards",
        "wise"
    )

    ADJECTIVE_SUFFIXES = (
        "ful",
        "less",
        "ous",
        "able",
        "ible",
        "ive",
        "al",
        "ic",
        "ish",
        "y",
        "ary",
        "ant",
        "ent"
    )

    NOUN_SUFFIXES = (
        "ness",
        "ment",
        "tion",
        "sion",
        "ity",
        "ship",
        "hood",
        "dom",
        "ism",
        "ist"
    )

    def __init__(self, irregular_nouns_path: str | None = None, irregular_verbs_path: str | None = None):

        data_directory = (
            Path(__file__).parent / "data"
        )

        if irregular_nouns_path is None:
            irregular_nouns_path = (
                data_directory / "irregular_nouns.json"
            )

        if irregular_verbs_path is None:
            irregular_verbs_path = (
                data_directory / "irregular_verbs.json"
            )

        self.irregular_nouns = (
            self._load_irregular_nouns(
                irregular_nouns_path
            )
        )

        self.irregular_verbs = (
            self._load_irregular_verbs(
                irregular_verbs_path
            )
        )

    @staticmethod
    def _load_irregular_verbs(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        irregular_verbs = {}

        for lemma, forms in data.items():
            lemma = lemma.casefold()
            irregular_verbs[lemma] = lemma

            for form in forms:
                irregular_verbs[form.casefold()] = lemma

        return irregular_verbs

    @staticmethod
    def _load_irregular_nouns(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        irregular_nouns = {}

        for lemma, form in data.items():
            irregular_nouns[form.casefold()] = lemma.casefold()
            irregular_nouns[lemma.casefold()] = lemma.casefold()

        return irregular_nouns

    def __noun_lemma(self, word: str) -> str:
        if word in self.irregular_nouns:
            return self.irregular_nouns[word]

        rules = [
            (r'([^aeiou])ies\b', r'\1y'),
            (r'(s|x|z|ch|sh)es\b', r'\1'),
            (r'([^aeiou])oes\b', r'\1o'),
            (r's\b', r'')
        ]

        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, word)

            if result != word:
                return result

        return word

    def __verb_lemma(self, word: str) -> str:
        if word in self.irregular_verbs:
            return self.irregular_verbs[word]

        if word.endswith("ies"):
            return re.sub(r'ies$', 'y', word)

        if re.search(r'(s|x|z|ch|sh)es$', word):
            return re.sub(r'es$', '', word)

        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]

        if word.endswith("ied"):
            return re.sub(r'ied$', 'y', word)

        if word.endswith("ed"):
            base = word[:-2]

            if base.endswith("e"):
                return base

            if (
                len(base) >= 2
                and base[-1] == base[-2]
                and base[-1] not in "aeiou"
            ):
                return base[:-1]

            return base

        if word.endswith("ing"):
            base = word[:-3]

            if (
                len(base) >= 2
                and base[-1] == base[-2]
                and base[-1] not in "aeiou"
            ):
                return base[:-1]

            if base.endswith("e"):
                return base

            return base

        return word

    def __adjective_lemma(self, word: str) -> str:
        if word in {"better", "best"}:
            return "good"

        if word in {"worse", "worst"}:
            return "bad"

        if word in {"less", "least"}:
            return "little"

        if word in {"more", "most"}:
            return "much"

        if word.endswith("est"):
            base = word[:-3]

            if (
                len(base) >= 2
                and base[-1] == base[-2]
                and base[-1] not in "aeiou"
            ):
                return base[:-1]

            return base

        if word.endswith("er"):
            base = word[:-2]

            if (
                len(base) >= 2
                and base[-1] == base[-2]
                and base[-1] not in "aeiou"
            ):
                return base[:-1]

            return base

        return word

    def __adverb_lemma(self, word: str) -> str:
        return word

    def __detect_pos(self, word: str) -> str:
        if word in self.PRONOUNS:
            return "PRON"

        if word in self.DETERMINERS:
            return "DET"

        if word in self.PREPOSITIONS:
            return "ADP"

        if word in self.CONJUNCTIONS:
            return "CONJ"

        if word in self.irregular_nouns:
            return "NOUN"

        if word in self.irregular_verbs:
            return "VERB"

        if word in self.AUXILIARIES:
            return "AUX"

        if word.endswith(self.ADVERB_SUFFIXES):
            return "ADV"

        if word.endswith(self.ADJECTIVE_SUFFIXES):
            return "ADJ"

        if word.endswith(self.NOUN_SUFFIXES):
            return "NOUN"

        if re.search(r'[^aeiou]ies$', word):
            return "NOUN"

        if re.search(r'(s|x|z|ch|sh)es$', word):
            return "NOUN"

        if re.search(r'[^aeiou]oes$', word):
            return "NOUN"

        if word.endswith("s") and not word.endswith("ss"):
            return "NOUN"

        if word.endswith(("ing", "ed")):
            return "VERB"

        return "UNKNOWN"

    def lemmatize(self, word: str, pos: str | None = None) -> str:
        word = word.casefold()

        if pos:
            pos = pos.upper()

            if pos == "NOUN":
                return self.__noun_lemma(word)

            if pos == "VERB":
                return self.__verb_lemma(word)

            if pos == "ADJ":
                return self.__adjective_lemma(word)

            if pos == "ADV":
                return self.__adverb_lemma(word)

            return word

        if word in self.PRONOUNS:
            return word

        if word in self.DETERMINERS:
            return word

        if word in self.PREPOSITIONS:
            return word

        if word in self.CONJUNCTIONS:
            return word

        pos = self.__detect_pos(word)

        if pos == "NOUN":
            return self.__noun_lemma(word)

        if pos == "VERB":
            return self.__verb_lemma(word)

        if pos == "ADJ":
            return self.__adjective_lemma(word)

        if pos == "ADV":
            return self.__adverb_lemma(word)

        return word

    def analyze(self, word: str) -> dict:
        word = word.casefold()
        pos = self.__detect_pos(word)
        lemma = self.lemmatize(word, pos)

        return {
            "word": word,
            "pos": pos,
            "lemma": lemma
        }

    def lemmatize_text(self, words: list[str]) -> list[str]:
        return [self.lemmatize(word) for word in words]