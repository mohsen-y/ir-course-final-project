import ast

from nltk.stem import WordNetLemmatizer

from schemas import SetOperationEnum


def and_op(*args) -> set:
    return set.intersection(*args)


def or_op(*args) -> set:
    return set.union(*args)


def not_op(docs: set, all_docs: set) -> set:
    return all_docs - docs


def evaluate_boolean_query(
    node: ast.expr,
    inverted_index: dict[str, list[int]],
    lemmatizer: WordNetLemmatizer,
    all_docs: set[int],
) -> list[int]:
    if isinstance(node, ast.Call):
        func = globals()[node.func.id]
        if node.func.id == SetOperationEnum.not_op.value:
            arg = evaluate_boolean_query(
                node.args[0], inverted_index, lemmatizer, all_docs
            )
            return func(arg, all_docs)
        else:
            args = [
                evaluate_boolean_query(arg, inverted_index, lemmatizer, all_docs)
                for arg in node.args
            ]
            return func(*args)

    elif isinstance(node, ast.Constant):
        return set(inverted_index.get(lemmatizer.lemmatize(node.s).lower(), []))

    else:
        raise ValueError("Unsupported boolean query structure")
