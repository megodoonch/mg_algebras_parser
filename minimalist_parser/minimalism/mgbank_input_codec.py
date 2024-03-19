"""
functions for getting from the original MGBank files in data/raw to terms and trees and other useful data structures.
also for getting from my own terms written to file an strings back to the terms they represent
"""

import json
import logging
import os
import sys

import nltk
import re

from minimalist_parser.minimalism.minimalist_algebra import MinimalistAlgebra
from ..algebras import algebra
from ..convert_mgbank.mgbank2algebra import lexical_constant_maker
from . import minimalist_algebra as mg
from . import movers
from .mg_types import MGType
from ..convert_mgbank.slots import slot_name2slot
from ..trees.trees import Tree
from .prepare_packages.triple_prepare_package import HMTriplesPreparePackages

# for NLTK tree reading
brackets = '()'
open_b, close_b = brackets
open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
node_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
leaf_pattern = '[^%s%s]+' % (open_pattern, close_pattern)

# for AlgebraTerm interpreting
OP_SEP = "+"
CONJ = "[+conj]"
MV_SEP = "."
PREP_SEP = "_"


def get_prepare(label_segment):
    # e.g. prepare_simple or just simple
    if label_segment.startswith("prepare"):
        prepare_parts = label_segment.split(PREP_SEP)
        return prepare_parts[1]
    else:
        return label_segment


def leaf_is_silent(label):
    # MGBank gives silent LIs names like [trans]
    return label.startswith('[')


def leaf_is_conjunction(label):
    # conjunctions are written to file with a CONJ marker since this affects what operations can apply
    return label.endswith(CONJ)


def is_adjoin(label):
    return label.endswith("+adjoin")


def get_from_slot(parts, mover_type):
    # e.g. A or A.1 (if ListMovers)
    if MV_SEP in parts[1]:  # ListMovers
        assert mover_type == mg.ListMovers
        split_slot = parts[1].split(MV_SEP)
        index = int(split_slot[1])
        from_slot = split_slot[0]
        return from_slot, int(index)
    else:
        return parts[1]


def constant_maker(leaf):
    """
    TODO we'll probably want to stop using this in favour of lexical_constant_maker
    A simple constant-making function, where the label is just the string
    representation of the leaf and the leaf itself is the "function"
    @param leaf: algebra object; here, Expression
    @return: AlgebraOp
    """
    return algebra.AlgebraOp(repr(leaf), leaf)


def label2minimalist_op(label: str, minimalist_algebra: MinimalistAlgebra):
    """
    Given a label, return an AlgebraOp.
    For internal nodes, return a MinimalistFunction.
        use inner_alg to get the inner_op and prepare functions
    For leaves, return an AlgebraOp with value Expression
        use the mover_type to give the right type of empty movers to the leaves
    Depends on the order of parts in the MInimalistFunction.name,
     and the separator is assumed to be +
     If there's an index, it's assumed to be separated by its slot with a '.'
     (separators defined in global variables)
    @param prepare_functions: PreparePackages (default HMTriplePreparePackages)
    @param label: str
    @param inner_alg: an HMAlgebra (default HMTripleAlgebra)
    @param mover_type: a Movers (sub)-type
    @return: AlgebraOp
    """
    # use HMTripleAlgebra by default
    if minimalist_algebra is None:
        logging.warning("No minimalist algebra given; using default with string triples")
        inner_alg = HMTripleAlgebra()
        prepare_functions = HMTriplesPreparePackages()
        minimalist_algebra = MinimalistAlgebra(inner_algebra=inner_alg, prepare_packages=prepare_functions)
    label = label.strip()
    parts = label.split(OP_SEP)
    inner_op = None
    prepare = None
    to_slot = None
    from_slot = None
    index = None
    if parts[0] == "merge1":
        # merge1+inner_op+prepare_blah
        try:
            min_op = minimalist_algebra.merge1
            inner_op = getattr(minimalist_algebra.inner_algebra, parts[1])
            prepare = getattr(minimalist_algebra.prepare_packages, get_prepare(parts[2]))
        except AttributeError as e:
            logging.error(e)
            logging.debug(f"parts: {parts}")
            raise e
    elif parts[0] == "merge2":
        # merge2+slot+inner_up+prepare_blah
        min_op = minimalist_algebra.merge2
        to_slot = slot_name2slot(parts[1])
        inner_op = getattr(minimalist_algebra.inner_algebra, parts[2])
        prepare = getattr(minimalist_algebra.prepare_packages, get_prepare(parts[3]))
    elif parts[0] == "move1":
        # move1+slot+inner_op
        min_op = minimalist_algebra.move1
        from_slot = get_from_slot(parts, minimalist_algebra.mover_type)
        if type(from_slot) == tuple:
            from_slot, index = from_slot
        inner_op = getattr(minimalist_algebra.inner_algebra, parts[2])
    elif parts[0] == "move2":
        # move2+from_slot+to_slot
        min_op = minimalist_algebra.move2
        from_slot = get_from_slot(parts, minimalist_algebra.mover_type)
        if type(from_slot) == tuple:
            from_slot, index = from_slot
        to_slot = slot_name2slot(parts[2])

    else:  # lexical item
        conj = leaf_is_conjunction(label)
        if conj:
            label = label[:-len(CONJ)]

        return lexical_constant_maker(label=label,
                                      inner_algebra=minimalist_algebra.inner_algebra,
                                      mover_type=minimalist_algebra.mover_type,
                                      conj=conj,
                                      silent=leaf_is_silent(label))
        # return minimalist_algebra.make_leaf(label, conj=conj, silent=leaf_is_silent(label))

    op = mg.MinimalistFunction(min_op,
                               inner_op=inner_op,
                               prepare=prepare,
                               to_slot=to_slot,
                               from_slot=from_slot,
                               index=index,
                               adjoin=is_adjoin(label))

    # assert op.name == label, f"new function {op} name isn't {label}."
    return op


def tree2term(tree: algebra.Tree, minimalist_algebra: MinimalistAlgebra):
    """
    Turns a Tree with nodes labelled in a way label2minimalist_function can
        understand into a term where the nodes are MinimalistFunctions with
        inner algebra inner_alg and mover store type mover_type
    @param prepare_functions:
    @param tree: Tree (as defined in trees.py)
    @param inner_alg: an HMAlgebra
    @param mover_type: Movers or a subtype
    @return: AlgebraTerm
    """
    if tree.children is None:
        return algebra.AlgebraTerm(label2minimalist_op(tree.parent, minimalist_algebra))
    else:
        kids = [tree2term(kid, minimalist_algebra) for kid in tree.children]
        return algebra.AlgebraTerm(label2minimalist_op(tree.parent, minimalist_algebra),
                                   kids)


def read_corpus_file(path):
    """
    Reads in MGBank corpus file (json format) and returns the 4th and 5th trees,
        which are the "plain" tree, with MGBank operation names, and the
        "annotated" tree, where each label is annotated with a partial result,
        minus the feature unification
    @param path: str: path to corpus. Assumes there are subfolders containing
                        the files and the files have .mrg suffixes
    @return: list of pairs of nltk.Trees
    """

    derivation_trees_as_strings = mgbank_corpus_file2tree_as_strings(path, 4)
    annotated_trees_as_strings = mgbank_corpus_file2tree_as_strings(path, 5)
    derivation_trees = [nltk.Tree.fromstring(tree,
                                             remove_empty_top_bracketing=True,
                                             node_pattern=node_pattern,
                                             leaf_pattern=leaf_pattern) for tree in derivation_trees_as_strings]
    annotated_trees = [nltk.Tree.fromstring(tree,
                                            remove_empty_top_bracketing=True,
                                            node_pattern=node_pattern,
                                            leaf_pattern=leaf_pattern) for tree in annotated_trees_as_strings]

    return list(zip(derivation_trees, annotated_trees))

    # with open(path) as f:
    #     raw_data = json.load(f)
    #
    # tree_pairs = []
    # for k in raw_data.keys():
    #     tree_pairs.append((nltk.Tree.fromstring(raw_data[k][4],
    #                                             remove_empty_top_bracketing=True,
    #                                             #node_pattern=node_pattern,
    #                                             #leaf_pattern=leaf_pattern
    #                                             ),
    #                        nltk.Tree.fromstring(raw_data[k][5],
    #                                             remove_empty_top_bracketing=True,
    #                                             #node_pattern=node_pattern,
    #                                             #leaf_pattern=leaf_pattern
    #                                             )))
    #
    # return tree_pairs


def read_plain_text_corpus_file(path: str) -> list[nltk.Tree]:
    """
    Reads in a file of trees, where each tree is on its own line
    @param path: str: path to corpus
    @return: list of nltk.Trees
    """

    with open(path) as f:
        trees = []
        for line in f.readlines():
            line = line.replace("  ", " ")  # remove double spaces
            line = line.replace("( ", "(")  # remove extra spaces around (,)
            line = line.replace(") ", ")")
            line = line.strip()
            # print(line)
            trees.append(nltk.Tree.fromstring(line,
                                              remove_empty_top_bracketing=True,
                                              # node_pattern=node_pattern,
                                              # leaf_pattern=leaf_pattern,
                                              )
                         )
    return trees


def read_tsv_corpus_file(path: str) -> list[tuple[str, nltk.Tree]]:
    """
    Reads in a file of trees, where each line is sentence \t tree
    @param path: str: path to corpus
    @return: list of pairs of (string, nltk.Trees)
    """
    with open(path) as f:
        lines = f.readlines()
        pairs = []
        for line in lines:
            parts = line.split("\t")
            tree_part = parts[1].strip()
            tree_part = tree_part.replace("  ", " ")
            tree_part = tree_part.replace("( ", "(")
            tree_part = tree_part.replace(") ", ")")
            # print(tree_part)
            tree = nltk.Tree.fromstring(tree_part,
                                        remove_empty_top_bracketing=True,
                                        # node_pattern=node_pattern,
                                        # leaf_pattern=leaf_pattern
                                        )
            pairs.append((parts[0], tree))

    return pairs


def mgbank_corpus_file2tree_as_strings(path, tree_number=0):
    """
    Reads in MGBank corpus file (json format) and returns the tree_numberth tree of each sentence
        default is 0, the full derivation tree with all features
    @param path: str: path to corpus. Assumes there are subfolders containing
                        the files and the files have .mrg suffixes
    @param tree_number: int: which tree to return (default 0)
                                0: the full derivation tree (default)
                                1: Xbar
                                2: with <,> arrows
                                3: internal nodes labeled with intermediate results (full features)
                                4. derivation tree with no unification-style subfeatures
                                5. internal nodes labeled with intermediate results (no unification-style subfeatures)
    @return: list of strings
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return [data[entry][tree_number] for entry in data]
    except FileNotFoundError:
        print("\n*********location:", os.getcwd(), file=sys.stderr)
        raise



def mgbank_corpus_file2nltk_trees(path, tree_number=0):
    """
    Reads in MGBank corpus file (json format) and returns the tree-numberth tree of each sentence as an NLTK tree

    @param path: str: path to corpus. Assumes there are subfolders containing
                        the files and the files have .mrg suffixes
    @param tree_number: int: which tree to return (default 0)
                                0: the full derivation tree (default)
                                1: Xbar
                                2: with <,> arrows
                                3: internal nodes labeled with intermediate results (full features)
                                4. derivation tree with no unification-style subfeatures
                                5. internal nodes labeled with intermediate results (no unification-style subfeatures)
    @return: list of nltk trees
    """

    trees_as_strings = mgbank_corpus_file2tree_as_strings(path, tree_number)
    return [nltk.Tree.fromstring(tree,
                                 remove_empty_top_bracketing=True,
                                 node_pattern=node_pattern,
                                 leaf_pattern=leaf_pattern) for tree in trees_as_strings]


if __name__ == "__main__":
    # s = "( move1+ABar+concat_left ( merge1+concat_right+prepare_simple ( [decl] )  ( move1+A+concat_left ( merge1+concat_right+prepare_simple ( [pres] )  ( merge2+A+concat_right+prepare_simple ( merge1+concat_right+prepare_prefix ( [trans] )  ( merge2+ABar+concat_right+prepare_simple ( says )  ( merge1+concat_right+prepare_simple ( [focalizer] )  ( merge1+concat_right+prepare_simple ( [decl] )  ( move1+A+concat_left ( merge1+concat_right+prepare_simple ( [past] )  ( merge2+A+concat_right+prepare_simple ( merge1+concat_right+prepare_prefix ( [trans] )  ( move1+A+concat_left ( move2+A.0+A ( merge1+concat_right+prepare_simple ( allowed )  ( merge1+concat_right+prepare_simple ( [decl] )  ( merge1+concat_right+prepare_simple ( to )  ( merge2+A+concat_right+prepare_simple ( merge1+concat_right+prepare_simple ( get )  ( merge1+concat_left+prepare_simple ( merge1+concat_right+prepare_simple ( and[+conj] )  ( stronger ) )  ( stronger ) ) )  ( him ) ) ) ) ) ) ) )  ( that ) ) ) ) ) ) ) )  ( he ) ) ) ) ) )"
    # tree = nltk.Tree.fromstring(s,
    #                             remove_empty_top_bracketing=True,
    #                             node_pattern=node_pattern,
    #                             leaf_pattern=leaf_pattern)
    # print(tree)

    from ..algebras.hm_triple_algebra import HMTripleAlgebra
    from ..minimalism.prepare_packages.triple_prepare_package import HMTriplesPreparePackages

    alg = HMTripleAlgebra()
    prepare_package = HMTriplesPreparePackages()

    predictions = read_plain_text_corpus_file("../../results/predictions/bart/one_prediction.txt")
    gold = read_tsv_corpus_file("../../data/processed/old/seq2seq/one_sentence.tsv")
    #
    gold_nltk_tree = gold[0][1]
    predicted_nltk_tree = predictions[0]

    print("sentence:", gold[0][0])

    # gold_nltk_tree.draw()
    # predicted_nltk_tree.draw()

    print("gold")
    gold_tree = Tree.nltk_tree2tree(gold_nltk_tree)
    gold_term = tree2term(gold_tree, None)
    gold_term.to_nltk_tree().draw()
    gold_expression = gold_term.evaluate()
    # print(gold_expression)
    gold_expression.inner_term.to_nltk_tree().draw()
    print(gold_expression.spellout())
    print()
    print("predicted")
    predicted_tree = Tree.nltk_tree2tree(predicted_nltk_tree)
    predicted_term = tree2term(predicted_tree, None)
    predicted_expression = predicted_term.evaluate()
    predicted_expression.inner_term.to_nltk_tree().draw()
    print(predicted_term.evaluate().spellout())

    # inner terms annotate nodes
    print(predicted_term.annotated_tree())

    # l = "<(_,_), (3,4)>:cH{}"
    # from hm_pair_algebra_mgbank import HMIntervalPairsAlgebra
    #
    # alg = HMIntervalPairsAlgebra()
    #
    # print(label2minimalist_function(l, alg))

    # trees = read_corpus_file(sys.argv[1])
    #
    # for plain, annotated in trees:
    #     print("plain")
    #     print(plain)
    #     print("\nannotated")
    #     print(annotated)
