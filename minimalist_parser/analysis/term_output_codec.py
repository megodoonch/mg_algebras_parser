"""
Transforms algebra terms into NLTK trees and LaTeX forests
"""


import nltk
import re
from minimalist_parser.algebras import algebra
from minimalist_parser.minimalism.movers import ListMovers

# VERBOSE = True
VERBOSE = False


def log(s):
    if VERBOSE:
        print(s)


brackets = '()'
open_b, close_b = brackets
open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
node_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
leaf_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
remove_empty_top_bracketing = True,


def term2nltk_tree(term: algebra.AlgebraTerm):
    """
    Given an algebra term, return an NLTK tree with the same node labels
    @param term: AlgebraTerm
    @return: nltk.Tree
    """
    if term.children is None:
        return nltk.Tree(term.parent.name, [])
    else:
        new_kids = [term2nltk_tree(kid) for kid in term.children]
        if "." in term.parent.name:
            log("\n**** found index ***\n")
            log(term.parent.name)
        return nltk.Tree(term.parent.name, new_kids)


def seq2seq_output2list_and_forest(tree_as_list: [str]):
    """
    Makes latex forest trees from the output of the seq2seq model
    Adds closing brackets at the end to make up enough to match all opening brackets.
    Doesn't add closing brackets in the middle if it would end the tree early
    @param tree_as_list: list of strings as output by seq2seq, e.g.
            ["(", "move1+ABar+concat_left", "(", "merge1+concat_right+prepare_prefix", "(", "[decl]", ")"]
    @return: string that can be copied into a LaTeX document, including \begin{forest} and \end{forest}
    """
    # put braces around each item
    tree_as_list = [f"{{{label}}}" for label in tree_as_list]
    # replace (,) with [,]
    new_tree_as_list = []
    opening = 0
    closing = 0
    for label in tree_as_list:
        if label == "{(}":
            new_tree_as_list.append("[")
            opening += 1
        elif label == "{)}":
            if closing < opening - 1:  # only add closing brackets if it wouldn't end the tree
                new_tree_as_list.append("]")
                closing += 1
        else:
            new_tree_as_list.append(label)
    # close any still open brackets
    diff = opening - closing
    new_tree_as_list += "]" * diff
    tree_as_string = " ".join(new_tree_as_list)
    special_characters = "#$%&~_^"
    for c in special_characters:
        tree_as_string = tree_as_string.replace(c, f"\\{c}")
    return new_tree_as_list, "\\begin{forest}" + tree_as_string + "\\end{forest}"


def file2list_and_forest(path, n):
    with open(path, 'r') as f:
        all_strings = f.readlines()
        tree_as_list = all_strings[n].split()
        return seq2seq_output2list_and_forest(tree_as_list)





if __name__ == "__main__":
    x = ["(", "move1+ABar+concat_left",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[decl]", ")",
         "(", "move1+A+concat_left",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[pres]", ")",
         "(", "merge2+A+concat_right+prepare_simple",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[trans]", ")",
         "(", "merge2+ABar+concat_right+prepare_simple",
         "(", "says", ")",
         "(", "merge1+concat_right+prepare_simple",
         "(", "[focalizer]", ")",
         "(", "merge1+concat_right+prepare_simple",
         "(", "[decl]", ")",
         "(", "move1+A+concat_left",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[pres]", ")",
         "(", "merge2+A+concat_right+prepare_simple",
         "(", "merge1+concat_right+prepare_simple",
         "(", "is", ")"]

    y = ["(", "move1+ABar+concat_left",
         ")", ")", ")",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[decl]", ")",
         "(", "move1+A+concat_left",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[pres]", ")",
         "(", "merge2+A+concat_right+prepare_simple",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[trans]", ")",
         "(", "merge2+ABar+concat_right+prepare_simple",
         "(", "says", ")",
         "(", "merge1+concat_right+prepare_simple",
         "(", "[focalizer]", ")",
         "(", "merge1+concat_right+prepare_simple",
         "(", "[decl]", ")",
         "(", "move1+A+concat_left",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[pres]", ")",
         "(", "merge2+A+concat_right+prepare_simple",
         "(", "merge1+concat_right+prepare_simple",
         "(", "is", ")"]

    z = ["(", "merge1+concat_right+prepare_simple",
         "(", "[decl]", ")",
         "(", "move1+A+concat_left",
         "(", "merge1+concat_right+prepare_prefix",
         "(", "[pres]", ")",
         "(", "merge2+A+concat_right+prepare_simple",
         "(", "merge1+concat_right+prepare_simple",
         "(", ":", ")",
         "(", "merge1+concat_right+prepare_simple",
         "(", "[det]", ")",
         "(", "merge1+concat_left+prepare_simple",
         "(", "merge1+concat_right+prepare_simple",
         "(", "num", ")", "(", "million", ")", ")",
         "(", "$", ")", ")", ")", ")",
         "(", "merge1+concat_right+prepare_simple",
         "(", "[det]", ")",
         "(", "merge1+concat_right+prepare_simple+adjoin",
         "(", "mr."]

    # print(seq2seq_output2forest(z))

    l, forest = file2list_and_forest('../../results/predictions/bart/predictions.txt', 0)
    print(forest)


    # from mgbank2algebra import label2minimalist_function
    from minimalist_parser.algebras.hm_triple_algebra import HMTripleAlgebra
    # from hm_pair_algebra_mgbank import HMIntervalPairsAlgebra
    from minimalist_parser.trees.transducer import nltk2tree
    from minimalist_parser.minimalism.mgbank_input_codec import tree2term, label2minimalist_op
    from minimalist_parser.minimalism.prepare_packages.triple_prepare_package import HMTriplesPreparePackages

    #
    # alg = HMIntervalPairsAlgebra()
    alg = HMTripleAlgebra()
    #
    leaf = algebra.AlgebraTerm(label2minimalist_op("hi", None))
    in_term = algebra.AlgebraTerm(label2minimalist_op("move1+A+concat_right", None), [leaf])
    print(in_term)
    #
    nltk_tree = term2nltk_tree(in_term)
    print(nltk_tree)
    #
    out_tree = nltk2tree(nltk_tree)
    print(out_tree)
    #
    out_term = tree2term(out_tree, minimalist_algebra=None)
    print(out_term)
    #
    # print(out_term == in_term)

    # print(out_term.evaluate())
