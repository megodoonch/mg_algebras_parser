silent_vocabulary = ["[past]", "[adjunctizer]", "[decl]", "[Q]"]
conjunctions = ["and", "or", "but", "to"]  # TODO extract these from the training set


def silent_head2id(token: str) -> int:
    return silent_vocabulary.index(token)


def id2silent_head(index: int) -> str:
    return silent_vocabulary[index]


def get_number_of_silent_heads():
    return len(silent_vocabulary)
