"""
Defines a general class of tree transducers
"""


from .trees import Tree
# from mgbank2algebra import split_label
# from mgbank_parser import EMPTY_INTERVAL_HEAD
import nltk

# logging: change VERBOSE to False to stop printing everything
from logger import logger
VERBOSE = False
#VERBOSE = True

def log(string):
    logger(string, VERBOSE)


# temporary states
q_lex = "q_lex"
Q = "Q"


class TransitionRule:
    def __init__(self, state_builder, tree_builder):
        """
        state_builder: function from state list and parent to (state, new parent)
        @param tree_builder: function from [parent, child0, ...] to tree
        """
        self.state_builder = state_builder
        self.tree_builder = tree_builder

    def apply(self, states, trees):
        """
        Applies the state and tree building functions and returns
         a pair of new state, new tree
        @param states: list of states
        @param trees: list of parent label::daughter trees
        @return: state, tree
        """
        log(f"applying to {states}, {trees[0]}")
        new_state, new_parent = self.state_builder(states, trees[0])
        log(f"got {new_state}, {new_parent}")
        if len(trees) > 1:
            remaining_trees = trees[1:]
        else:
            remaining_trees = []
        return new_state, self.tree_builder([new_parent] + remaining_trees)


def transduce(t: Tree, rule_generator, d=None):
    """
    given term in algebra 1, returns state and term of algebra 2
    @param d: a dict for looking stuff up if needed. e.g.
                for getting sentence indices for tree addresses
    @param rule_generator: function from a tree and list of states
                    to a TransitionRule
    @param t: Tree to transduce
    @return: new Tree
    """
    if t.children is None:
        return rule_generator(t, [], d).apply([], [t.parent])

    else:
        outs = [transduce(t.children[x], rule_generator, d) for x in range(len(t.children))]
        states = [state for state, _ in outs]
        trees = [tree for _, tree in outs]
        rule = rule_generator(t, states, d)
        return rule.apply(states, [t.parent] + trees)


def nltk2tree(nltk_tree: nltk.Tree):
    """
    Get a Tree from an nltk.Tree
    Assumes the leaves are nltk.Trees as well
    @param nltk_tree: nltk.Tree
    @return: Tree
    """
    if type(nltk_tree) == str:
        return Tree(nltk_tree)
    elif nltk_tree.height() == 1:
        return Tree(nltk_tree.label())
    else:
        new_kids = [nltk2tree(c) for c in nltk_tree]
        return Tree(nltk_tree.label(), new_kids)


if __name__ == "__main__":

    pass

    # from mgbank2algebra import t as der_tree
    #
    # # print(dt)
    # #
    # # der_tree = nltk2tree(dt)
    #
    # # print(der_tree)
    #
    # s = der_tree.evaluate().spellout()
    # print(s)
    # d = {}
    # for i, w in enumerate(s.split()):
    #     address = w.split(".")[1]
    #     d[address] = i
    #
    # Q, new_t = transduce(der_tree, address2index_rule_generator, d)
    # print(new_t)
    #
    # expr = new_t.evaluate()
    #
    # print(expr)
    #
    # print(expr.is_complete())
