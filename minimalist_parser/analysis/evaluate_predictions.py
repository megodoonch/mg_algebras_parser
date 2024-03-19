"""
some evaluations on predicted trees
"""
import os
from collections import Counter
from datetime import datetime

from minimalist_parser.algebras.algebra import AlgebraOp, AlgebraError, AlgebraTerm
from minimalist_parser.algebras.hm_triple_algebra import HMTripleAlgebra
from minimalist_parser.convert_mgbank.mgbank2training_data import tsv2sentences_and_terms, text_file2terms
from minimalist_parser.minimalism.mg_errors import MGError
from minimalist_parser.analysis.to_vulcan import *
from minimalist_parser.minimalism.minimalist_algebra import leaves_in_order, leaf_spellouts_in_order, MinimalistAlgebra
from nltk.translate import bleu
import logging

import sys

from minimalist_parser.minimalism.movers import DSMCMovers, ListMovers

# logging
logdir = "/home/meaghan/PycharmProjects/minimalist_parser/logfiles/analysis/bart/official_split/"
log_file = "predictions.log"
# logdir = "./"
os.makedirs(logdir, exist_ok=True)

root = logging.getLogger()
root.setLevel(logging.DEBUG)  # this seems to set the minimum logging level

# log to standard out
stream_handler = logging.StreamHandler(sys.stdout)
# choose how much you want to print to stout (DEBUG is everything, ERROR is almost nothing)
# handler.setLevel(logging.DEBUG)
# handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.WARNING)
# handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
root.addHandler(stream_handler)

file_handler = logging.FileHandler(filename=f"{logdir}/{log_file}")
# choose how much you want to print to log file (DEBUG is everything, ERROR is almost nothing)
file_handler.setLevel(logging.DEBUG)
# handler.setLevel(logging.INFO)
# handler.setLevel(logging.WARNING)
# handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(levelname)s %(message)s')
file_handler.setFormatter(formatter)
root.addHandler(file_handler)


#
# # overridden in script
# # VERBOSE = True
# VERBOSE = False
#
#
# def log(s):
#     if VERBOSE:
#         print(s)
#


class NLTKError(Exception):
    """Exception raised for NLTK tree reading problems
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = message



def complete_trees(predictions):
    incomplete = []
    for tree in predictions:
        expr = tree.evaluate()
        complete = expr.is_complete()

        if not complete:
            logging.info("not complete")
            logging.info(f"{expr}")
        incomplete += tree

    return incomplete


def is_error(x):
    return issubclass(type(x), Exception)


def compute_precision_recall_f1_from_counters(pred_counter: Counter, gold_counter: Counter):
    """
    Author: Jonas Groschwitz.
    Compute P, R ,F, use when everything for each set has been put into the same counter.
    @param pred_counter: Counter for predictions
    @param gold_counter: Counter for gold
    @return: P, R, F
    """
    total_predictions = sum(pred_counter.values())
    total_gold = sum(gold_counter.values())
    true_predictions = 0
    for key in pred_counter.keys():
        true_predictions += min(pred_counter[key], gold_counter[key])
    return compute_precision_recall_f1(true_predictions, total_predictions, total_gold)


def compute_precision_recall_f1_from_counter_lists(pred_counters: list[Counter], gold_counters: list[Counter]):
    """
    Author: Jonas Groschwitz.
    Compute P, R ,F. Use when items to be compared are spread over a lit of counters
    @param pred_counters: list of counters for predictions
    @param gold_counters: list of counters for gold
    @return: P, R, F
    """
    total_gold, total_predictions, true_predictions = compute_correctness_counts_from_counter_lists(gold_counters,
                                                                                                    pred_counters)
    return compute_precision_recall_f1(true_predictions, total_predictions, total_gold)


def compute_correctness_counts_from_counter_lists(gold_counters, pred_counters):
    total_predictions = sum([sum(c.values()) for c in pred_counters])
    total_gold = sum([sum(c.values()) for c in gold_counters])
    true_predictions = 0
    for pred_counter, gold_counter in zip(pred_counters, gold_counters):
        for key in pred_counter.keys():
            true_predictions += min(pred_counter[key], gold_counter[key])
    return total_gold, total_predictions, true_predictions


def compute_precision_recall_f1(true_predictions, total_predictions, total_gold):
    if total_predictions > 0:
        precision = true_predictions / total_predictions
    else:
        precision = 0
    recall = 0 if total_gold == 0 else true_predictions / total_gold
    if true_predictions == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


from ..minimalism.prepare_packages.triple_prepare_package import HMTriplesPreparePackages


def get_terms_from_tsv_and_check_interpretability(in_path) \
        -> tuple[list[tuple[int, str, AlgebraTerm]], int, list[tuple[int, str, AlgebraTerm]]]:
    """
    given tsv filepath, reads in all sentences and terms and checks them for interpretability and completeness.
    A term is interpretable if its evaluate method works, and is complete if there are no leftover movers
    @param in_path:
    @return: interpretable terms, int (number of uninterpretable or incomplete terms)
    """
    minimalist_algebra = MinimalistAlgebra(inner_algebra=HMTripleAlgebra(), prepare_packages=HMTriplesPreparePackages(),
                                           mover_type=ListMovers)
    logging.info(f"reading in terms from TSV {in_path}...")
    triples = tsv2sentences_and_terms(in_path, minimalist_algebra=minimalist_algebra, brackets="()")
    logging.info("done")
    errors = [t for t in triples if is_error(t[2])]
    print(f"{len(triples)} total items in {in_path}")
    if len(errors) > 0:
        logging.warning(f"{len(triples) - len(errors)} out of {len(triples)} items in file interpreted as terms")
        logging.warning(f"{len(errors)} errors in reading in file")
    else:
        logging.info("No errors")
    for i, e, in errors:
        logging.debug(f"\t{i}. {type(e).__name__} {e}")
    good_terms = []

    errors = []

    # check the terms we read in for completeness and errors
    for i, sentence, term in triples:
        if term is None:
            # to keep everything in order, lines that couldn't be interpretted as a term have None in place of a term
            errors.append((i, NLTKError("Couldn't get term from file")))
            continue
        try:
            expression = term.evaluate()
            c = expression.is_complete()
            good_terms.append((i, sentence, term))
            if not c:
                logging.warning("\n *** not complete ***")
                logging.warning(f"{i} {sentence}")
        except Exception as err:
            errors.append((i, err, sentence))
    if len(errors) > 0:
        logging.warning(f"{len(errors)} errors in interpreting terms read in from {in_path}.")
        for i, e, s in errors:
            logging.debug(f"\t{i}. {e} in {s}")
    else:
        logging.info("No errors in interpreting terms")
    return good_terms, len(triples) - len(good_terms), triples


def get_terms_from_text_file_and_check_interpretability(in_path) \
        -> tuple[list[tuple[int, AlgebraTerm]], int, list[tuple[int, AlgebraTerm]]]:
    """
    given filepath containing terms, reads in all terms and checks them for interpretability and completeness.
    A term is interpretable if its evaluate method works, and is complete if there are no leftover movers
    @param in_path:
    @return: interpretable terms, int (number of uninterpretable or incomplete terms)
    """
    minimalist_algebra = MinimalistAlgebra(inner_algebra=HMTripleAlgebra(), prepare_packages=HMTriplesPreparePackages(),
                                           mover_type=ListMovers)
    logging.info(f"reading in terms from {in_path}...")
    terms = text_file2terms(in_path, minimalist_algebra=minimalist_algebra, brackets="()")
    logging.info("done")
    errors = [t for t in terms if is_error(t[1])]
    print(f"{len(terms)} total items in {in_path}")
    if len(errors) > 0:
        logging.warning(f"{len(terms) - len(errors)} out of {len(terms)} items in file interpreted as terms")
        logging.warning(f"{len(errors)} errors in reading in file")
        for i, e, in errors:
            logging.debug(f"\t{i}. {type(e).__name__} {e}")

    good_terms = []

    errors = []

    # check the terms we read in for completeness and errors
    # if trying to get the term got us an error, term is actually an error
    for i, term in terms:
        if is_error(term):
            # to keep everything in order, lines that couldn't be interpreted as a term have their error
            # in place of a term
            errors.append((i, NLTKError("Couldn't get term from file")))
            continue
        try:
            expression = term.evaluate()
            c = expression.is_complete()
            good_terms.append((i, term))
            if not c:
                logging.warning("not complete")
                logging.warning(f"{i}")
        except TypeError as err:
            raise err
        except Exception as err:
            errors.append((i, err))

    if len(errors) > 0:
        logging.warning(f"{len(errors)} errors in interpreting terms read in from {in_path},"
                        f" including errors from reading in terms.")
        structural = [error for error in errors if type(error[1]) in [AlgebraError, NLTKError]]
        logging.info(f"{len(structural)} structural errors")
        logging.info(f"{len(errors) - len(structural)} other errors")
        for i, e, in errors:
            logging.debug(f"\t{i}. {type(e).__name__} {e}")
    else:
        logging.info(f"No errors in interpreting terms read in from {in_path}")
    return good_terms, len(terms) - len(good_terms), terms


def spell_out_terms_and_compare_to_sentences(sentences: list[str], terms: list[AlgebraTerm]):
    """
    given a list of (index, sentence, term), spells out the term and compares to the sentence
    @param terms: list of AlgebraTerms to evaluate
    @param sentences: list of sentences as strings
    @return: list of incorrect (index, term, gold sentence, generated sentence), list of errors
    """
    errors = []
    incorrect = []
    all_generated_sentences = []
    print()
    logging.warning("Terms that evaluate to the wrong string:")
    for i, (s, t) in enumerate(zip(sentences, terms)):
        compare_spellout_to_sentence(i, s, t, errors, incorrect, all_generated_sentences)

    print()
    logging.error(f"{len(errors)} errors while interpreting terms")
    logging.error(f"{len(incorrect)} terms did not evaluate to the correct sentence")
    return incorrect, errors, all_generated_sentences


def compare_spellout_to_sentence(i, sentence, term, errors, incorrect, all_generated_sentences):
    """
    given a term, spell it out and compare it to the given sentence. Update errors and incorrect.
    @param errors: list of errors
    @param i: index of sentence/term
    @param incorrect: list of incorrect terms
    @param sentence: sentence to compare spellout to
    @param term: AlgebraTerm over minimalist algebra
    @return: None, just updates
    """
    try:
        expression = term.evaluate()
        inner_term = expression.inner_term
        try:
            generated_sentence = inner_term.evaluate().spellout()
            if generated_sentence != sentence:
                logging.info(f"\n{i}\ngold\t{sentence}\npred\t{generated_sentence}")
                incorrect.append((i, term, sentence, generated_sentence))
            all_generated_sentences.append(generated_sentence)
        except Exception as e:
            logging.error(f"Error interpreting inner term of sentence {i}. {sentence}:\n {e}\n")
            errors.append((i, sentence, e))
            all_generated_sentences.append(repr(e))

    except Exception as e:
        logging.debug(f"{i}. not interpretable")
        # logging.error(f"Error interpreting minimalist term of sentence {i}. {sentence}:\n {e}\n")
        errors.append((i, sentence, e))
        all_generated_sentences.append(repr(e))


def compare_predictions_to_gold(path_to_gold, path_to_predictions, pickle_path=None):
    """
    reads in gold and predicted trees and runs various analyses. Pickles outputs if pickle_path given.
        - checks interpretability
        - checks for exact match of predicted trees (untested)
        - interprets the terms and spells out the string, checking for correctness
        - If pickling, creates:
            - all.pickle: all gold and all predictions, as GMA terms
            - wrong_string.pickle: predicted GMA and inner terms and predicted and gold strings for terms with wrong string interpretation
            - wrong_tree.pickle: gold and all predictions, as GMA terms, for incorrect term predictions
            - correct.pickle: predicted GMA and inner terms and predicted and gold strings for correct terms

    @param path_to_gold: path to the gold TSV file.
    @param path_to_predictions: path to the predicted txt file.
    @param pickle_path: path to the directory to put pickles in.
    @return: all gold triples (index, sentence, term), all predicted pairs (index, term)
    """
    # Get the terms from the files and check their interpretability
    _, n_bad_gold, all_gold = get_terms_from_tsv_and_check_interpretability(path_to_gold)
    _, n_bad_predicted, all_predicted = get_terms_from_text_file_and_check_interpretability(path_to_predictions)

    print()
    logging.error(f"{n_bad_predicted} total uninterpretable predictions")

    # check for exact match
    different = [i for ((i, sentence, gold_term), (j, predicted_term))
                 in zip(all_gold, all_predicted) if gold_term != predicted_term]
    correct = [i for ((i, sentence, gold_term), (j, predicted_term))
               in zip(all_gold, all_predicted) if gold_term == predicted_term]

    logging.info(f"{len(correct)} predictions are exactly correct")
    logging.debug(f"\tindices: {correct}")

    # compare the spellout of the predicted terms with the sentence
    gold_sentences = [s for _, s, _ in all_gold]
    predicted_terms = [t for _, t in all_predicted]
    incorrect, errors, all_generated_sentences = spell_out_terms_and_compare_to_sentences(gold_sentences, predicted_terms)
    assert len(all_generated_sentences) == len(predicted_terms), f"{len(predicted_terms)} predicted terms and {len(all_generated_sentences)} generated sentences"

    if pickle_path:

        # all
        gold_and_predicted_terms2pickle(all_gold, all_predicted, all_generated_sentences, pickle_path=pickle_path + "all.pickle")

        # wrong string
        term_and_sentences2pickle(incorrect, pickle_path + "wrong_string.pickle")

        # wrong tree
        different_gold = [triple for triple in all_gold if triple[0] in different]
        different_predicted = [pair for pair in all_predicted if pair[0] in different]
        different_sentences = [all_generated_sentences[i] for i in range(len(all_generated_sentences)) if i in different]
        gold_and_predicted_terms2pickle(different_gold, different_predicted, different_sentences, pickle_path + "wrong_tree.pickle")
        gold_for_wrong_tree = [(i, gold_tree, gold_sentence, generated_sentence) for
                               ((i, gold_sentence, gold_tree), generated_sentence)
                               in zip(different_gold, different_sentences)]
        term_and_sentences2pickle(gold_for_wrong_tree, pickle_path=pickle_path +"wrong_tree_gold.pickle")

        # right tree
        if len(correct) > 0:
            strings = [triple[1] for triple in all_gold if triple[0] in correct]
            correct_predicted = [pair for pair in all_predicted if pair[0] in correct]
            correct = [(i, term, sentence, sentence) for ((i, term), sentence) in zip(correct_predicted, strings)]
            term_and_sentences2pickle(correct, pickle_path + "correct.pickle")
        else:
            print()
            logging.warning("no correct trees to pickle")

        print("\nPickles in", pickle_path)

    return all_gold, all_predicted


def derivation_tree_subtree_string_yields(term: AlgebraTerm):
    """
    Given a term, returns a list of pairs of parents and the labels of the leaves, in order, that they dominate
    @param term: AlgebraTerm
    @return: (AlgebraOp, string tuple) list
    """
    tuples = []

    def derivation_tree_subtree_string_yields_inner(t: AlgebraTerm):

        tuples.append((t.parent, tuple(leaves_in_order(t))))
        if not t.is_leaf():
            for child in t.children:
                derivation_tree_subtree_string_yields_inner(child)

    try:
        derivation_tree_subtree_string_yields_inner(term)
    except AttributeError as e:
        return []
    return tuples


def derivation_tree_subtree_string_yields_ignore_silent(term: AlgebraTerm):
    """
    Given a term, returns a list of pairs of parents and the labels of the leaves, in order, that they dominate,
        but without any silent heads
    @param term: AlgebraTerm
    @return: (AlgebraOp, string tuple) list
    """

    tuples = []

    def derivation_tree_subtree_string_yields_inner(t: AlgebraTerm):
        if t.is_leaf():
            interp = t.evaluate().spellout()
            if len(interp) > 0:
                tuples.append((interp, tuple(leaf_spellouts_in_order(t))))
        else:
            kids = [w for w in leaf_spellouts_in_order(t) if len(w) > 0]
            tuples.append((t.parent, tuple(kids)))
            for child in t.children:
                derivation_tree_subtree_string_yields_inner(child)

    try:
        derivation_tree_subtree_string_yields_inner(term)
    except AttributeError as e:
        return []
    return tuples


def derivation_tree_subtree_string_yields_internal_nodes(term: AlgebraTerm):
    """
    Given a term, returns a list of pairs of nonterminal parents
     and the labels of the leaves, in order, that they dominate
    @param term: AlgebraTerm
    @return: (AlgebraOp, string list) list
    """

    tuples = []

    def derivation_tree_subtree_string_yields_inner(t: AlgebraTerm):
        if not t.is_leaf():
            kids = [w for w in leaves_in_order(t) if len(w) > 0]
            tuples.append((t.parent, tuple(kids)))
            for child in t.children:
                derivation_tree_subtree_string_yields_inner(child)

    try:
        derivation_tree_subtree_string_yields_inner(term)
    except AttributeError as e:
        return []
    return tuples


def get_rules_from_term(term: AlgebraTerm) -> list[tuple[AlgebraOp, tuple[AlgebraOp]]]:
    """
    Gets all "rules" -- parent and tuple of children's roots -- from a term.
    Use tuples so they are hashable for the Counter needed for P,R,F
    @param term: AlgebraTerm
    @return: list of pairs, (parent, roots of children)
    """
    rules = []
    # some things in our list might be errors instead of terms
    if is_error(term):
        return rules

    def get_rules_from_term_inner(t: AlgebraTerm):
        if not t.is_leaf():
            rules.append((t.parent, tuple([c.parent for c in t.children])))
            for c in t.children:
                get_rules_from_term_inner(c)

    get_rules_from_term_inner(term)
    return rules


def slices(term: AlgebraTerm) -> list[tuple[AlgebraOp, tuple[AlgebraOp]]]:
    """
    Given a term, get all slices: heads paired with the operations they control
    @param term: AlgebraTerm over a MinimalistAlgebra
    @return: (head, list of operations) list
    """
    if is_error(term):
        return []
    tuples = []

    def inner(subterm: AlgebraTerm):
        """
        Get the slice for this subterm and add (head, operations) to tuples
        """
        slice = []

        def inner_most(subsubterm: AlgebraTerm):
            """
            if the root of this subtree is lexical, it's the head. The slice is done; add it to tuples.
            otherwise, add the operation to the slice and continue toward the head
            """
            if subsubterm.children is None:
                # we're at the head! Add to tuples.
                tuples.append((subsubterm.parent, tuple(slice)))
            elif len(subsubterm.children) == 1:
                # add move node and continue
                slice.append(subsubterm.parent)
                inner_most(subsubterm.children[0])
            else:
                # add merge node
                slice.append(subsubterm.parent)
                # continue right toward head
                inner_most(subsubterm.children[0])
                # new slice(s) on left (should just be one, but could be a mistake in tree structure)
                for child in subsubterm.children[1:]:
                    inner(child)
        inner_most(subterm)
    inner(term)
    return tuples


def get_c_commands_of_leaves(term:AlgebraTerm):
    c_commanded = []
    if is_error(term) or term.children is None:
        return []

    def inner(subterm: AlgebraTerm):
        for i in range(len(subterm.children)):
            if subterm.children[i].is_leaf():
                c_commanded.append((subterm.children[i].parent, tuple(subterm.children)))
            else:
                inner(subterm.children[i])

    inner(term)
    return c_commanded


def get_c_commanded_leaves_of_leaves(term: AlgebraTerm):
    c_commanded = []

    if is_error(term) or term.children is None:
        return []
    def inner(subterm: AlgebraTerm):
        for i in range(len(subterm.children)):
            if subterm.children[i].is_leaf():
                leaves = [leaves_in_order(subterm.children[j]) for j in range(len(subterm.children)) if j != i]
                flattened_leaves = [leaf for leaf_list in leaves for leaf in leaf_list]
                c_commanded.append((subterm.children[i], tuple(flattened_leaves)))
            else:
                inner(subterm.children[i])

    inner(term)
    return c_commanded





def corpus2p_r_f(gold, predicted, function, print_out=False):
    """
    Given the gold and predicted items, get the things to compare using function and return the P,R,F
    @param gold: gold items (e.g. trees)
    @param predicted: predicted items (e.g. trees)
    @param function: function from two such items to iterables of things to compare
    @return: P, R, F
    """

    gold = [function(gold_term) for gold_term in gold]
    predicted = [function(pred_term) for pred_term in predicted]
    gold_counter = Counter()
    for g in gold:
        gold_counter.update(g)
    predicted_counter = Counter()
    for p in predicted:
        predicted_counter.update(p)
    if print_out:
        print("gold")
        for entry in gold_counter:
            print(gold_counter[entry], entry)
        print("predicted")
        for entry in predicted_counter:
            print(predicted_counter[entry], entry)
    p, r, f = compute_precision_recall_f1_from_counters(gold_counter=gold_counter, pred_counter=predicted_counter)
    print(f"P: {round(p, 4)}")
    print(f"R: {round(r, 4)}")
    print(f"F: {round(f, 4)}")
    return p, r, f


def main(gold_path, pred_path, pickle_path=None):
    """
    Runs a bunch of analyses and makes vulcan-readable pickles of relevant terms if given a pickle_path.
    For ease of running as a script from the command line, update this function with whatever you want to do differently -- treat it as
        the script portion of the file. If you want it to print more/less to the logfile or to the console,
        change the logger settings at the top of evaluate_predictions.py.
    Currently:
        - compares predictions to gold and prints full results to logfiles/analysis/bart/predictions.log and less full
            results to the console.
        - calculates several metrics of tree similarity and prints them to the log and console.
    @param gold_path: path from whereever you are running the script from to the gold file (TSV string \t tree).
    @param pred_path: path from wherever you are running the script from to the predictions (just trees).
    @param pickle_path: path from wherever you are running the script to the directory where you want the
            pickles written to. Default None, which means no pickles are created.
    """

    logging.info("")
    logging.info("")
    logging.info("**** New run ****")
    logging.info(datetime.now())

    if pickle_path is not None:
        os.makedirs(pickle_path, exist_ok=True)

    all_gold, all_predicted = compare_predictions_to_gold(gold_path, pred_path, pickle_path)

    # _, _, all_gold = get_terms_from_tsv_and_check_interpretability(to_predict)
    # _, _, all_predicted = get_terms_from_text_file_and_check_interpretability(pred_path)

    print("\nEVALUATIONS")
    logging.info("EVALUATIONS")

    sample = -1

    gold_terms = [tup[2] for tup in all_gold][:sample]
    predicted_terms = [tup[1] for tup in all_predicted][:sample]

    print("\nrules")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, get_rules_from_term)
    logging.info("rules")
    logging.info(f"P: {p}, R: {r}, F: {f}")

    print("\nsubtrees and the leaves they dominate")
    logging.info("For every node n, find all terminal nodes in the subtree n is the root of, in order. "
                 "Compare the multiset of pairs (parent node, sequence of leaves in subtree)")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, derivation_tree_subtree_string_yields)
    logging.info("subtrees and the leaves they dominate")
    logging.info(f"P: {p}, R: {r}, F: {f}")

    print("\nsubtrees ignore silent")
    logging.info("Just like subtrees and the leaves they dominate, except silent leaves like [decl] are ignored")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, derivation_tree_subtree_string_yields_ignore_silent)
    logging.info("subtrees ignore silent")
    logging.info(f"P: {p}, R: {r}, F: {f}")

    print("\nsubtrees internal nodes")
    logging.info("Just like subtrees and the leaves they dominate, except we only"
                 " consider non-terminal nodes, i.e. operations")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, derivation_tree_subtree_string_yields_internal_nodes)
    logging.info("subtrees internal nodes only")
    logging.info(f"P: {p}, R: {r}, F: {f}")


    print("\nslices")
    logging.info("Slices of derivation tree: heads and the operations they control")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, slices, print_out=False)
    logging.info("slices")
    logging.info(f"P: {p}, R: {r}, F: {f}")

    print("\nc-commanded subtrees")
    logging.info("C-commanded subtrees of heads")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, get_c_commands_of_leaves, print_out=False)
    logging.info("c-command")
    logging.info(f"P: {p}, R: {r}, F: {f}")

    print("\nc-commanded leaves")
    logging.info("C-commanded leaves of heads")
    p, r, f = corpus2p_r_f(gold_terms, predicted_terms, get_c_commanded_leaves_of_leaves, print_out=False)
    logging.info("c-command")
    logging.info(f"P: {p}, R: {r}, F: {f}")



if __name__ == "__main__":

    # NOTE this can ONLY be run from the command line if the arguments are also given from the command line, otherwise
    # python can't find the files. But it can b run in Pycharm at least with these.
    to_predict = "data/processed/seq2seq/sampled/official/to_predict.tsv"
    pred_path = "results/predictions/bart/official/predicted.txt"
    pickle_path = "results/analysis/bart/official_split/"


    if len(sys.argv) == 4:
        to_predict = sys.argv[1]
        pred_path = sys.argv[2]
        pickle_path = sys.argv[3]

    main(to_predict, pred_path, pickle_path)

