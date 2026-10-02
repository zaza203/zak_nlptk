import pandas as pd
import numpy as np
from collections import Counter
from .normalizer import TextNormalizer
from .segmentation import Segmentation

class TfIdF:
    def __init__(self):
        self.vocabulary_ = None
        self.idf_ = None
        self.tokenized_docs = None

    def _tokenize(self, documents):
        return [
            [
                token 
                for token in Segmentation(
                    TextNormalizer(lowercase=True).normalize(doc)
                ).word_segmentation()
                if token not in {".", ",", "?"}
            ]
            for doc in documents
        ]

    def fit(self, documents):
        self.tokenized_docs = self._tokenize(documents)
        term_counts = pd.DataFrame(
            [Counter(tokens) for tokens in self.tokenized_docs]
        ).fillna(0)

        self.vocabulary_ = term_counts.columns

        n = len(term_counts)
        df = (term_counts > 0).sum(axis=0)

        self.idf_ = 1 + np.log(n / df)

        return self

    def transform(self, documents):
        if self.vocabulary_ is None or self.idf_ is None:
            raise RuntimeError(
                "TfIdF has not been fitted. Call fit() first."
            )
        tokenized_docs = self._tokenize(documents)

        term_counts = pd.DataFrame(
            [Counter(tokens) for tokens in tokenized_docs]
        ).fillna(0)

        term_counts = term_counts.reindex(
            columns=self.vocabulary_,
            fill_value=0
        )

        tf = term_counts.div(
            term_counts.sum(axis=1),
            axis=0
        )

        tfidf = tf * self.idf_

        return tfidf

    def fit_transform(self, documents):
        self.fit(documents)

        return self.transform(documents)

    
    def cost_dist(self, v1, v2):
        dot = np.dot(v1, v2)
        denominator = np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

        return 1 - dot / denominator

    def distance_matrix(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]

        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distances[i, j] = self.cost_dist(matrix[i], matrix[j])

        return distances