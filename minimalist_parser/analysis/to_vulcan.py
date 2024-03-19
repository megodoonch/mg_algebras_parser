"""
functions for creating Vulcan-readable pickles.
Vulcan is a visualisation tool
"""

import nltk
from vulcan.pickle_builder.pickle_builder import PickleBuilder
from vulcan.data_handling.format_names import FORMAT_NAME_NLTK_TREE, FORMAT_NAME_STRING, FORMAT_NAME_TOKEN

from minimalist_parser.algebras.algebra import AlgebraTerm
from minimalist_parser.minimalism.mgbank_input_codec import read_plain_text_corpus_file, read_tsv_corpus_file


def gold_and_predicted_files2pickle(gold_path, prediction_path, pickle_path):
    """
    Given paths to TSV of gold sentence, term and txt file of terms, writes a vulcan-readable pickle of NLTK trees
        containing gold, predicted, and string.
    @param gold_path: path to gold TSV.
    @param prediction_path: path to predictions txt.
    @param pickle_path: what to call and where to write the pickle file.
    @return:
    """

    predictions = read_plain_text_corpus_file(prediction_path)
    gold = read_tsv_corpus_file(gold_path)
    gold = [(i, sentence, tree) for (i, (sentence, tree)) in enumerate(gold)]

    gold_and_predicted_nltk_trees2pickle(gold, predictions, pickle_path)


def gold_and_predicted_nltk_trees2pickle(gold, predictions, pickle_path):
    """
    Given gold pairs (sentence, NLTK tree) and predictions as NLTK trees,
    @param gold: list of pairs (sentence, tree)
    @param predictions: list of trees
    @param pickle_path: where to write the pickle
    @return:
    """

    pickler = PickleBuilder({"gold": FORMAT_NAME_NLTK_TREE,
                             "gold string": FORMAT_NAME_STRING,
                             "predicted string": FORMAT_NAME_STRING,
                             "predicted": FORMAT_NAME_NLTK_TREE,
                             "n": FORMAT_NAME_STRING}
                            )

    for (i, sentence, gold_tree), (pred_s, predicted_tree) in zip(gold, predictions):
        pickler.add_instances_by_name({
            "gold": gold_tree,
            "gold string": sentence,
            "predicted string": pred_s,
            "predicted": predicted_tree,
            "n": str(i)
        })

    pickler.write(pickle_path)


def gold_and_predicted_terms2pickle(gold: list[tuple[int, str, AlgebraTerm]], predictions: list[tuple[int, AlgebraTerm]]
                                    , all_predicted_sentences, pickle_path):
    """
    Given gold and predictions as AlgebraTerms, create a vulcan-readable pickle
    @param gold:
    @param predictions:
    @param pickle_path:
    @return:
    """
    new_gold = [(i, s, term_or_error2nltk_tree(term_or_error)) for (i, s, term_or_error) in gold]
    new_predictions = [(s, term_or_error2nltk_tree(term_or_error)) for (s, (i, term_or_error)) in zip(all_predicted_sentences, predictions)]
    gold_and_predicted_nltk_trees2pickle(new_gold, new_predictions, pickle_path)


def term_or_error2nltk_tree(term_or_error):
    """
    Given a term or an error, makes an NLTK tree
    If error, makes a single node labeled with the error
    @param term_or_error: either an AlgebraTerm or an Exception
    @return: nltk.Tree
    """
    if issubclass(type(term_or_error), Exception):
        tree = nltk.Tree.fromstring(f"({type(term_or_error).__name__})")
    elif type(term_or_error) == AlgebraTerm:
        tree = term_or_error.to_nltk_tree()
    else:
        raise TypeError(f"{term_or_error} should be AlgebraTerm or Exception, but is {type(term_or_error)}")
    return tree


def term_and_sentences2pickle(incorrect, pickle_path):
    """
    given a list of (probably incorrect) predictions, creates a vulcan-readable pickle
    @param incorrect: (index, predicted term, gold sentence, predicted sentence) list
    @param pickle_path: full path to pickle name
    """
    pickler = PickleBuilder({"gold sentence": FORMAT_NAME_STRING,
                             "generated": FORMAT_NAME_STRING,
                             "predicted GMA term": FORMAT_NAME_NLTK_TREE,
                             "predicted inner term": FORMAT_NAME_NLTK_TREE,
                             "n": FORMAT_NAME_STRING
                             })
    for i, predicted_tree, gold_sentence, generated_sentence in incorrect:
        try:
            inner_term = predicted_tree.evaluate().inner_term
        except Exception as e:
            inner_term = e
        inner_tree = term_or_error2nltk_tree(inner_term)


        pickler.add_instances_by_name({
            "gold sentence": gold_sentence,
            "generated": generated_sentence,
            "predicted GMA term": predicted_tree.to_nltk_tree(),
            "predicted inner term": inner_tree,
            "n": str(i)
        })
    pickler.write(pickle_path)
