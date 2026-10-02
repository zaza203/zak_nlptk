from collections import Counter
import numpy as np
import pickle

from .tf_idf import TfIdF


class NaiveBayes:

    def __init__(self, alpha: float = 1.0, feature_type: str = "count"):

        if feature_type not in {"count", "tfidf"}:
            raise ValueError("feature_type must be 'count' or 'tfidf'")

        if alpha < 0:
            raise ValueError("alpha must be greater than or equal to 0")

        self.alpha = alpha
        self.feature_type = feature_type

        self.class_priors = {}
        self.word_likelihoods = {}
        self.vocabulary = set()
        self.tfidf = None

    def fit(self, documents, labels):

        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length")

        if not documents:
            raise ValueError("documents cannot be empty")

        self.tfidf = TfIdF()

        class_counts = Counter(labels)

        total_documents = len(labels)

        self.class_priors = {
            class_name: count / total_documents
            for class_name, count in class_counts.items()
        }

        # ----------------------------------------------------------
        # COUNT-BASED NAIVE BAYES
        # ----------------------------------------------------------

        if self.feature_type == "count":

            self.tfidf.fit(documents)

            self.vocabulary = set(self.tfidf.vocabulary_)

            class_word_counts = {class_name: Counter() for class_name in class_counts}

            for tokens, label in zip(self.tfidf.tokenized_docs, labels):
                class_word_counts[label].update(tokens)

            vocabulary_size = len(self.vocabulary)

            self.word_likelihoods = {}

            for class_name in class_counts:

                total_words = sum(class_word_counts[class_name].values())

                denominator = total_words + self.alpha * vocabulary_size

                self.word_likelihoods[class_name] = {}

                for word in self.vocabulary:

                    word_count = class_word_counts[class_name][word]

                    probability = (word_count + self.alpha) / denominator

                    self.word_likelihoods[class_name][word] = probability

        # ----------------------------------------------------------
        # TF-IDF-BASED NAIVE BAYES
        # ----------------------------------------------------------

        else:

            X = self.tfidf.fit_transform(documents)

            self.vocabulary = set(X.columns)

            self.word_likelihoods = {}

            for class_name in class_counts:

                class_indices = [
                    i for i, label in enumerate(labels) if label == class_name
                ]

                # Sum TF-IDF weights for this class
                class_weights = X.iloc[class_indices].sum(axis=0)

                total_weight = class_weights.sum()

                vocabulary_size = len(self.vocabulary)

                denominator = total_weight + self.alpha * vocabulary_size

                self.word_likelihoods[class_name] = {}

                for word in self.vocabulary:

                    weight = class_weights[word]

                    probability = (weight + self.alpha) / denominator

                    self.word_likelihoods[class_name][word] = probability

        return self

    def predict_proba(self, documents):

        if not self.class_priors:
            raise RuntimeError("NaiveBayes has not been fitted. " "Call fit() first.")

        if self.feature_type == "count":

            tokenized_docs = self.tfidf._tokenize(documents)

            probabilities = []

            for tokens in tokenized_docs:

                word_counts = Counter(tokens)

                log_scores = {}

                for class_name in self.class_priors:

                    log_score = np.log(self.class_priors[class_name])

                    for word, count in word_counts.items():

                        if word not in self.vocabulary:
                            continue

                        likelihood = self.word_likelihoods[class_name][word]

                        log_score += count * np.log(likelihood)

                    log_scores[class_name] = log_score

                max_log = max(log_scores.values())

                scores = {
                    class_name: np.exp(score - max_log)
                    for class_name, score in log_scores.items()
                }

                total = sum(scores.values())

                probabilities.append(
                    {
                        class_name: (scores[class_name] / total)
                        for class_name in self.class_priors
                    }
                )

            return probabilities

        else:

            X = self.tfidf.transform(documents)

            probabilities = []

            for _, row in X.iterrows():

                log_scores = {}

                for class_name in self.class_priors:

                    log_score = np.log(self.class_priors[class_name])

                    for word in self.vocabulary:

                        weight = row[word]

                        if weight == 0:
                            continue

                        likelihood = self.word_likelihoods[class_name][word]

                        log_score += weight * np.log(likelihood)

                    log_scores[class_name] = log_score

                max_log = max(log_scores.values())

                scores = {
                    class_name: np.exp(score - max_log)
                    for class_name, score in log_scores.items()
                }

                total = sum(scores.values())

                probabilities.append(
                    {
                        class_name: (scores[class_name] / total)
                        for class_name in self.class_priors
                    }
                )

            return probabilities

    def predict(self, documents):

        probabilities = self.predict_proba(documents)

        return [max(probability, key=probability.get) for probability in probabilities]

    # ----------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------

    def _calculate_metrics(self, documents, labels):

        if len(documents) != len(labels):
            raise ValueError("documents and labels must have " "the same length")

        predictions = self.predict(documents)

        classes = sorted(set(labels) | set(predictions), key=str)

        class_to_index = {class_name: index for index, class_name in enumerate(classes)}

        matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for actual, predicted in zip(labels, predictions):

            i = class_to_index[actual]
            j = class_to_index[predicted]

            matrix[i, j] += 1

        accuracy = sum(
            prediction == label for prediction, label in zip(predictions, labels)
        ) / len(labels)

        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": matrix,
            "classes": classes,
            "predictions": predictions,
        }

        return metrics

    def evaluate(self, documents, labels):

        metrics = self._calculate_metrics(documents, labels)

        classes = metrics["classes"]
        matrix = metrics["confusion_matrix"]

        precision = {}
        recall = {}
        f1 = {}

        for class_name in classes:

            index = classes.index(class_name)

            tp = matrix[index, index]

            fp = matrix[:, index].sum() - tp

            fn = matrix[index, :].sum() - tp

            if tp + fp == 0:
                class_precision = 0.0
            else:
                class_precision = tp / (tp + fp)

            if tp + fn == 0:
                class_recall = 0.0
            else:
                class_recall = tp / (tp + fn)

            if class_precision + class_recall == 0:
                class_f1 = 0.0
            else:
                class_f1 = (
                    2
                    * class_precision
                    * class_recall
                    / (class_precision + class_recall)
                )

            precision[class_name] = class_precision
            recall[class_name] = class_recall
            f1[class_name] = class_f1

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1

        return metrics

    # ----------------------------------------------------------
    # SAVE / LOAD
    # ----------------------------------------------------------

    def save(self, path):

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):

        with open(path, "rb") as file:
            return pickle.load(file)
