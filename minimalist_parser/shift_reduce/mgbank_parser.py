"""
Parser for MGBank style MGs

Inner algebra: head movement pairs a la Milos or maybe me

SpIC in effect

Types: lexical, conjunction, must HM

"""
from copy import deepcopy

from minimalist_parser.algebras.algebra_objects.triples import Triple
from minimalist_parser.algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from minimalist_parser.algebras.algebra_objects.intervals import Interval, Pair
from minimalist_parser.algebras.hm_triple_algebra  import HMTripleAlgebra
from minimalist_parser.minimalism.minimalist_algebra import MinimalistAlgebra
from minimalist_parser.convert_mgbank.slots import A, Abar
from minimalist_parser.minimalism.minimalist_algebra import Expression, MinimalistFunction
from minimalist_parser.minimalism.movers import Slot, Movers, ListMovers
from minimalist_parser.minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare

VERBOSE = True
# VERBOSE = False



def log(string):
    if VERBOSE:
        print(string)


# make the algebras
pair_alg = HMIntervalPairsAlgebra()
triple_alg = HMTripleAlgebra()
pair_prepare = IntervalPairPrepare()
EMPTY_INTERVAL_HEAD = pair_alg.empty_leaf


# global variables: slot names
# Note that the A to_slot is complex and handles the EPP stuff in the final
# version of the DSMC in John's thesis


# for testing
def string2interval_expressions(string: str, mover_type=Movers):
    """
    makes Expressions out of a string, separated by whitespace
    everything is by default -conj
    @param mover_type: Movers or another Movers subtype
    @param string: str
    @return: list of Expressions
    """
    index = 0
    es = []
    for _ in string.split(" "):
        es.append(Expression(pair_alg.make_leaf(Interval(index)), mover_type()))
        index += 1
    return es


def string2string_expressions(string: str, mover_type=Movers):
    """
    makes Expressions out of a string, separated by whitespace
    everything is by default -conj
    @param mover_type: type; must be subtype of Movers
    @param string: str
    @return: list of Expressions
    """
    index = 0
    es = []
    for word in string.split(" "):
        es.append(Expression(triple_alg.make_leaf(word), mover_type()))
        index += 1
    return es


class ParserState:
    """
    Minimalist Algebra mg state
    No features, just Expressions
    Currently encodes full sentence, stack just with intervals, actions
        just as function and stack indices applied to
    Note there is no buffer: everything starts on the stack
    Attributes:
        actions: list of actions already taken,
         as list of (MinimalistFunction, list of 1 or 2 indices) pairs
        stack: list of Expressions
        sentence: original string, to be separated on whitespace
    """

    def __init__(self, minimalist_algebra: MinimalistAlgebra, sentence: str, stack: [Expression] = None,
                 actions: [(MinimalistFunction, [int])] = None):
        """
        Creates an instance of the state of the mg
        If stack and actions are None, this is for initialising the mg,
            with stack = expressions of all the intervals, and actions = []
        @param sentence: str
        @param stack: list of Expressions, or None to initialise
        @param actions: list of actions already taken,
         as list of (MinimalistFunction, list of 1 or 2 indices) pairs, or None
        """
        self.minimalist_algebra = minimalist_algebra
        self.actions = actions
        if actions is None:
            self.actions = []
        self.sentence = sentence
        self.stack = stack
        if stack is None:
            self.stack = self.string2expressions(sentence)

    def string2expressions(self, string: str):
        """
        makes Expressions out of a string, separated by whitespace
        everything is by default -conj
        @param string: str
        @return: list of Expressions with the inner terms over the inner algebra of self.minimalist_algebra
        """
        es = []
        for index, word in enumerate(string.split()):
            if type(self.minimalist_algebra.inner_algebra) == HMIntervalPairsAlgebra:
                word = Interval(index)
            es.append(self.minimalist_algebra.make_leaf(word))
        return es

    def apply_function(self, function: MinimalistFunction, functor_index,
                       other_index=None):
        """
        Applies given function to functor and other
        @param function: MinimalistFunction: the function to use
        @param functor_index: int: index in the stack of the functor, 0-indexed.
                                    -1 for new empty head
        @param other_index: int: index in the stack of the argument, 0-indexed.
                                    -1 for new empty head
        @return: ParserState
        """
        assert functor_index < len(self.stack) and \
               (other_index is None or other_index < len(self.stack))

        new_actions = self.actions.copy()
        new_stack = self.stack.copy()

        if other_index is None:
            # Move
            assert function.mg_op.__name__ in {MinimalistAlgebra.move1.__name__, MinimalistAlgebra.move2.__name__}
            new_item = function.function([self.stack[functor_index]])
            new_stack[functor_index] = new_item
            new_actions.append((function, [functor_index]))
        else:
            # Merge
            assert function.mg_op.__name__ in {MinimalistAlgebra.merge1.__name__, MinimalistAlgebra.merge2.__name__}

            # check for empty heads (indices -1)
            # pop the items from the stack if found
            if functor_index >= 0:
                functor = self.stack[functor_index]
            else:
                functor = self.minimalist_algebra.inner_algebra.empty_item
            if other_index >= 0:
                other = self.stack[other_index]
            else:
                other = self.minimalist_algebra.inner_algebra.empty_item

            # apply the function
            new_item = function.function([functor, other])

            # if the head isn't empty, put the new item in its place
            if functor_index >= 0:
                new_stack[functor_index] = new_item
            else:  # if head is empty, append new item
                new_stack.append(new_item)
            if other_index >= 0:  # remove the other item, unless it's empty
                new_stack.pop(other_index)

            new_actions.append((function, [functor_index, other_index]))

        return ParserState(self.minimalist_algebra, self.sentence, new_stack, new_actions)

    def __repr__(self):
        string = "\nStack:\n"
        for index, expression in enumerate(self.stack):
            string += f"{index}:\t{expression}\n"
        string += "\nActions:\n"
        for index, action in enumerate(self.actions):
            string += f"{index}. {action[0]}({action[1]})\n"

        return string


if __name__ == "__main__":

    mg = MinimalistAlgebra(pair_alg, prepare_packages=pair_prepare)
    # mg = MinimalistAlgebra(triple_alg)

    mg.add_merge1(mg.inner_algebra.concat_right, mg.prepare_packages.suffix)
    mg.add_merge1(mg.inner_algebra.concat_right, mg.prepare_packages.prefix)
    mg.add_merge1(mg.inner_algebra.concat_right, mg.prepare_packages.simple)
    mg.add_merge1(mg.inner_algebra.concat_left, mg.prepare_packages.simple)

    for slot in mg.slots:
        for inner_op in [mg.inner_algebra.concat_right, mg.inner_algebra.concat_left]:
            mg.add_op(MinimalistFunction(mg.merge2, inner_op=inner_op, to_slot=slot))



    print(mg)


    # number_of_ops = 0
    # for opset in mgbank_operations_dict:
    #     print(opset, ":")
    #     print(mgbank_operations_dict[opset])
    #     number_of_ops += len(mgbank_operations_dict[opset])
    #
    # print("Number of ops:", number_of_ops)
    #
    s = "What did Mary read"
    print("\nSentence to parse:", s)
    #
    s1 = ParserState(mg, s)
    print("s1", s1)
    #
    # s2 = s1.apply_function(mg_abar, 3, 0)
    # print("s2", s2)
    #
    # s3 = s2.apply_function(mg_a, 2, 1)
    # print("s3",s3)
    #
    # s4 = s3.apply_function(mv_a, 1)
    # print("s4", s4)
    #
    # s5 = s4.apply_function(mg_r, -1, 1)
    # print("s5",s5)
    #
    # s6 = s5.apply_function(hm_suf_r, 0, 1)
    # print("s6",s6)
    #
    # s7 = s6.apply_function(mv_abar, 0)
    # print("s7",s7)
    #
    #
    # exprs = string2interval_expressions(s)
    # print("\nInitial state of mg:")
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    #
    # read, what = exprs[3], exprs[0]
    # read_what = mg.merge2(Slot(Abar), read, what, pair_alg.concat_right)
    #
    # print("interpret read what")
    # print(read_what.inner_term.evaluate())
    #
    # print("State of mg after merging items 3 and 0, storing wh-mover:")
    # exprs = exprs[1:3]
    # exprs.append(read_what)
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    #
    # mary = exprs[1]
    # vp = mg.merge2(Slot("A", a=True), read_what, mary, pair_alg.concat_right)
    #
    # print("interpret mary read what")
    # print("term:", vp.inner_term)
    # print(vp.inner_term.evaluate())
    #
    # print("State of mg after merging items 2 and 1, storing subject"
    #       " (needs case):")
    # exprs = [exprs[0], vp]
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    #
    # t = Expression(pair_alg.empty_item, movers=Movers())
    #
    # print("State of mg after merging a silent Tense head with item 1:")
    # t_bar = mg.merge1(pair_alg.concat_right, t, vp)
    # exprs = [exprs[0], t_bar]
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    #
    # print("State of mg after applying Move to the subject Mary in item 1:")
    # tp = mg.move1(A, t_bar, inner_op=pair_alg.concat_left)
    # exprs = [exprs[0], tp]
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    #
    # print("State of mg after Merging items 0 and 1, with T-to-R movement:")
    # c_bar = mg.merge1(pair_alg.concat_right, exprs[0], tp, prepare=pair_prepare.suffix)
    # exprs = [c_bar]
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    # #
    # print("State of mg after Moving wh-phrase to its final position:")
    # cp = mg.move1(Abar, c_bar, pair_alg.concat_left)
    # exprs = [cp]
    # for i, w in enumerate(exprs):
    #     print(f"{i}:\t{w}")
    # print()
    #
    # print("interpret outcome")
    # result = exprs[0]
    # inner_term = result.inner_term
    #
    # print(result.spellout())


    #
    # print("*** Trying to apply an illegal operation:")
    # exprs = string2interval_expressions(s)
    #
    # x = merge1(concat_left, exprs[0], exprs[3])
    # print(x)
    # y = merge1(concat_right, x, exprs[2])
    #
    # s = "who does Mary love and Sue hate"
    # print(s)
    # print("Requires excorporation and ATB head and phrase movement")
    #
    # exprs = string2interval_expressions(s)
    #
    # print(exprs)
    #
    # EMPTY_INTERVAL_HEAD = Expression(Item(empty_inner))
    #
    # who = exprs[0]
    # does = exprs[1]
    # mary = exprs[2]
    # love = exprs[3]
    # conjunction = exprs[4]
    # sue = exprs[5]
    # hate = exprs[6]
    #
    # # set and's conj feature to True
    # conjunction.inner_item.type.conj = True
    #
    # hate_who = merge2(Abar, hate, who)
    # print("\n1. Merge hate and who, but store who as an A-bar mover:", hate_who)
    #
    # love_who = merge2(Abar, love, who)
    # print("\n2. Merge love and who, but store who as an A-bar mover:", love_who)
    #
    # sue_hate_who = merge2(A, hate_who, sue)
    # print("sue hate who:", sue_hate_who)
    #
    # mary_love_who = merge2(A, love_who, mary)
    # print("mary love who:", mary_love_who)
    #
    # tbar_sue = merge1(concat_right, EMPTY_INTERVAL_HEAD, sue_hate_who)
    # print("T'(T, sue hate who)", tbar_sue)
    #
    # tp_sue = move1(A, tbar_sue)
    # print("TP(sue, hate who)", tp_sue)
    #
    # tbar_mary = merge1(concat_right, EMPTY_INTERVAL_HEAD, mary_love_who)
    # print("T'(T, mary love who)", tbar_mary)
    #
    # tp_mary = move1(A, tbar_mary)
    # print("TP(mary, love who)", tp_mary)
    #
    # coord_bar = merge1(excorporation, conjunction, tp_sue)
    # print("and sue hate who", coord_bar)
    #
    # coordP = merge1(hm_atb, coord_bar, tp_mary)
    # print("mary love and sue hate who", coordP)
    #
    # c_bar = merge1(hm_suffix, does, coordP)
    # print("does mary love and sue hate who", c_bar)
    #
    # cp = move1(Abar, c_bar)
    # print("who does mary love and sue hate", cp)
    #
    # print("\n These should report but not raise errors")
    # merge1(concat_right, mary, mary)
    #
    # move1(A, mary)
    #
    # print("\n Covert movement testing")
    #
    # cov = merge2(A, mary, does, covert=True, inner_op=concat_left)
    # print(cov)
    #
    # print(cov.inner_item.collapse())
    #
    #
