import json


class Codec:

    def __init__(self):

        self.token_to_id = {}
        self.id_to_token = {}

    def build(self, vocabulary):

        sorted_vocabulary = sorted(vocabulary)

        self.token_to_id = {
            token: index
            for index, token in enumerate(sorted_vocabulary)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

    def encode(self, tokens: list[str]) -> list[int]:

        ids = []

        for token in tokens:

            if token not in self.token_to_id:
                raise ValueError(
                    f"Unknown token: {token}"
                )

            ids.append(
                self.token_to_id[token]
            )

        return ids

    def decode(self, ids: list[int]) -> list[str]:

        tokens = []

        for index in ids:

            if index not in self.id_to_token:
                raise ValueError(
                    f"Unknown token ID: {index}"
                )

            tokens.append(
                self.id_to_token[index]
            )

        return tokens

    def save(self, path: str):

        data = {
            "token_to_id": self.token_to_id
        }

        with open(path, "w", encoding="utf-8") as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path: str):

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.token_to_id = {
            token: int(index)
            for token, index in data["token_to_id"].items()
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }