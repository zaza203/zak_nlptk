from collections import defaultdict
import random


def train_test_split(
    documents,
    labels,
    test_size=0.2,
    random_state=None
):
    if len(documents) != len(labels):
        raise ValueError(
            "documents and labels must have the same length"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1"
        )

    if random_state is not None:
        random.seed(random_state)

    class_data = defaultdict(list)

    for document, label in zip(documents, labels):
        class_data[label].append(document)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for label, class_documents in class_data.items():

        random.shuffle(class_documents)

        test_count = int(len(class_documents) * test_size)

        test_documents = class_documents[:test_count]
        train_documents = class_documents[test_count:]

        X_test.extend(test_documents)
        y_test.extend([label] * len(test_documents))

        X_train.extend(train_documents)
        y_train.extend([label] * len(train_documents))

    # Shuffle again so classes aren't grouped together
    train_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))

    random.shuffle(train_data)
    random.shuffle(test_data)

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return (
        list(X_train),
        list(X_test),
        list(y_train),
        list(y_test)
    )