"""
Minimalist Algebra in which inner items are inner algebra terms

"""
import sys
from copy import deepcopy

from minimalist_parser.minimalism.prepare_packages.prepare_packages_hm import ATBError
from ..algebras.algebra import AlgebraOp, AlgebraTerm, Algebra
from ..algebras.algebra_objects.intervals import Interval
from ..algebras.hm_algebra import HMError
from ..convert_mgbank.slots import Abar, A, R, Self, E
from ..minimalism.mg_types import MGType
from ..minimalism.mg_errors import *

from ..algebras.hm_triple_algebra import HMTripleAlgebra
from ..algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from ..minimalism.movers import Slot, Movers, ListMovers
from ..minimalism.prepare_packages.prepare_packages import PreparePackages
from ..trees.trees import Tree


import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug

# VERBOSE = True
VERBOSE = False




# make the algebras
triple_alg = HMTripleAlgebra()


# global variables: slot names
# Note that the A to_slot is complex and handles the EPP stuff in the final
# version of the DSMC in John's thesis

class Expression:
    """
    parse item
    Like an MG expression, but with no features

    Attributes:
        inner_term: hm_pair_algebra_mgbank.Item: the main structure
        type: just copied over from the inner_item for convenience
        movers: dict from to_slot names (see global variable SLOTS) to Intervals
    """

    def __init__(self, inner_term: AlgebraTerm or None, movers: Movers, mg_type: MGType = None):
        """
        Creates an instance of a minimalist algebra expression
        Type comes from the Item
        @param inner_term: Item, forming the inner algebra object
        @param movers: {str : Interval} dict. keys must be in SLOTS
        """
        self.inner_term = inner_term
        self.movers = movers
        self.mg_type = mg_type
        if mg_type is None:
            self.mg_type = MGType()

    def __repr__(self):
        return f"{self.inner_term}{self.mg_type}{self.movers}"

    def __eq__(self, other):
        return type(self) == type(other) and self.inner_term == other.inner_term and self.mg_type == other.mg_type \
            and self.movers == other.movers

    def equal_modulo_conj(self, other):
        return type(self) == type(other) and self.inner_term == other.inner_term and self.mg_type.lexical == other.mg_type.lexical \
            and self.movers == other.movers

    def is_lexical(self):
        """
        True if straight from the lexicon
        @return: bool
        """
        return self.mg_type.lexical

    def has_movers(self):
        """
        True if any mover slots are filled
        @return: bool
        """
        return len(self.movers.mover_dict) > 0

    def spellout(self):
        """
        Apply inner algebra's spellout operation to the root node's inner term
        @return:
        """
        if self.has_movers():
            raise MGError(f"Can't spell out an expressions containing movers: {self}")
        inner = self.inner_term.evaluate()
        try:
            return inner.spellout()
        except AttributeError:
            return inner

    def is_complete(self):
        """
        True if there are no movers and spellout is defined for inner item
        @return: bool
        """
        if self.has_movers():
            log("has movers")
            log(self.movers)
            return False
        try:
            self.spellout()
            return True
        except MGError as err:
            log(f"couldn't spell out {self}: {err}")
            return False

    def object_tuple_yield(self):
        """
        Make an expression-like tuple of the interpretations of the inner term and the movers
        # TODO test
        @return: inner object, Movers containing inner objects
        """
        main_object = self.inner_term.evaluate()
        movers = type(self.movers)()
        for slot in self.movers.mover_dict:
            mover = self.movers.mover_dict[slot]
            if type(self.movers) == ListMovers:
                movers_in_slot = [m.evaluate() for m in mover]
                movers.add_mover(Slot(slot, movers_in_slot, multiple=True))
            else:
                mover_in_slot = mover.evaluate()
                movers.add_mover(Slot(slot, mover_in_slot))
        return main_object, movers


def prepare_allowed(functor: Expression, prepare_name):
    if prepare_name == "excorporation":
        return functor.is_lexical() and functor.mg_type.conj
    elif prepare_name == "hm_atb":
        return functor.mg_type.conj and not functor.is_lexical()
    elif prepare_name in {"suffix", "prefix"}:
        # the conj case is for the undocumented l/r_merge_lex cases which can be repeated
        return functor.is_lexical() or functor.mg_type.conj
    else:
        return True


class MinimalistFunction(AlgebraOp):
    """
    A minimalist function of MG1/2 or MV1/2 types
    Attributes:
        function: a function from a list of 1 or 2 expressions to an expression
                    One of merge1, merge2, move1, move2 with the exact function
                     determined by the rest of the arguments to __init__
        from_slot: for Move: where the mover comes from
        index: for Move: if using ListMovers, which index to
                            take from a mover list
        to_slot: for Merge2 and Move2: where the mover gets stored
        prepare: a prepare function from the inner algebra from a pair of items
                    to a pair of items
        adjoin: bool: true if this is an adjoin function
    """

    def __init__(self, function, name=None, inner_op=None,
                 from_slot=None, to_slot=None,
                 index=None, prepare=None, adjoin=False, reverse=False):
        """
        Creates a function from a list of 1 or 2 expressions to an expression
        All parameters but function can be None
        function is pretty much the whole thing, but we store its properties
        @param function: merge1, merge2, move1, or move2
        @param inner_op: inner algebra function of an HMAlgebra
        @param from_slot: str: a SLOT (for MV1 and MV2)
        @param to_slot: str: a SLOT (for MG2 and MV2)
        @param index: int: just for ListMovers, indicates which mover in the
                        from_slot list to take out of storage
        @param prepare: inner algebra prepare function from an HMAlgebra
        @param adjoin: bool: whether this is an adjoin operation. Doesn't affect anything here,
                                but seems like useful info to pass along
        """
        super().__init__(name, function)  # these will get replaced
        self.from_slot = from_slot
        self.to_slot = to_slot
        self.index = index
        self.inner_op = inner_op
        self.mg_op = function
        self.prepare = prepare
        self.adjoin = adjoin
        self.mg_function_type = function

        # check we have the info we need
        if function.__name__ in {MinimalistAlgebra.merge1.__name__,
                        MinimalistAlgebra.move1.__name__} and self.inner_op is None:
            raise MGError("Merge1 and Move1 require an inner_op")

        if function.__name__ in {MinimalistAlgebra.merge2.__name__, MinimalistAlgebra.move2.__name__} and self.to_slot is None:
            raise MGError("Merge2 and Move2 require a to_slot")

        if function.__name__ in {MinimalistAlgebra.move1.__name__, MinimalistAlgebra.move2.__name__} and self.from_slot is None:
            raise MGError("Move requires a from_slot")

        # make the function
        if function.__name__ == MinimalistAlgebra.merge1.__name__:
            self.function = \
                lambda args: function(inner_op, args[0], args[1],
                                      prepare=prepare, adjoin=adjoin, reverse=reverse)
        elif function.__name__ == MinimalistAlgebra.merge2.__name__:
            self.function = lambda args: function(to_slot, args[0],
                                                  args[1],
                                                  prepare=prepare, adjoin=adjoin)
        elif function.__name__ == MinimalistAlgebra.move1.__name__:
            self.function = lambda args: function(from_slot, args[0],
                                                  inner_op, index=index, reverse=reverse)
        elif function.__name__ == MinimalistAlgebra.move2.__name__:
            self.function = lambda args: function(from_slot, to_slot,
                                                  args[0], index=index)
        else:
            raise ExistenceError(f"No such function as {function.__name__}")

        if name is None:
            # build operation name
            name = function.__name__
            if from_slot:
                name += f"+{from_slot}"
            if index is not None:
                name += f".{index}"
            if to_slot:
                name += f"+{to_slot.name}"
            if inner_op is not None:
                try:
                    name += f"+{inner_op.__name__}"
                except AttributeError:
                    name += f"+{inner_op.name}"
            if prepare is not None:
                name += f"+{prepare.__name__}"
            if self.adjoin:
                name += "+adjoin"
        self.name = name



    def __repr__(self):
        return self.name

    def properties(self):
        return f"name:\t{self.name}\ninner_op:\t{self.inner_op}\nprepare:\t{self.prepare}\nto_slot:\t{self.to_slot}\n" \
               f"from_slot:\t{self.from_slot}\nadjoin:\t{self.adjoin}"


class MinimalistAlgebra(Algebra):
    """
    Generalised Minimalist Algebras implement the structure-building behavious of Minimalist Grammars.
    Structures are built with the `inner algebra`, for example an algebra over strings, trees, or string tuples.
    Movers are stored in a Movers object, essentially a dictionary from storage slot names to movers.

    The objects of a MinimalistAlgebra are Expressions, which are essentially pairs:
        (term of inner algebra, movers). Expressions also have types that constrain operations that can apply.
        Expressions can be +/-lexical and +/-conj (coordinators)

    Operations belong to the categories Merge (binary) and Move (unary).
        Merge 1: uses an operation of the inner algebra to combine the two inner terms
        Merge 2: stores the second argument in a mover slot
        Move 1: takes a mover out of storage and combines it with the inner term using an operation of the inner algebra
        Move 2: switches a mover to a different mover slot

    Attributes:
         inner_algebra: Algebra used to build the actual output structures.
         mover_type: a subtype of Movers. There are a variety of ways to store movers. See movers.py.
         prepare_packages: PreparePackages object;
                            essentially pairs of tree homomorphisms that update the terms for, e.g., head movement.
        slots: list of slot names for moving. Default uses A, Abar, R, Self. If ListMovers is the movertype, also uses E
    """

    def __init__(self, inner_algebra: Algebra,
                 mover_type=Movers,
                 prepare_packages: None or PreparePackages or list[PreparePackages] = None,
                 slots: None or list[str] = None):
        """
        Initialise a generalised minimalist algebra with the given inner algebra, prepare packages, and mover type.
        @param inner_algebra: the algebra for building the actual output items.
        @param mover_type: how to store the movers.
        @param prepare_packages: PreparePackages: set of pairs of tree homomorphisms for minimally updating the
                terms before combining them. Mostly used for head movement.
        @param slots: optional list of mover slot names for auto-generating operations if desired.
                        Default [A, Abar, R, Self]
        """
        super().__init__()
        if prepare_packages is None:
            self.prepare_packages = [None]  # for easier looping in auto-generating functions
        else:
            self.prepare_packages = prepare_packages
        self.inner_algebra = inner_algebra
        self.name = "Minimalist Algebra"
        self.name += f" over {self.inner_algebra.name}"
        self.mover_type = mover_type
        self.add_constant_maker(self._constant_maker)
        # default slots
        # TODO this probably isn't right yet. some slots have special properties by default.
        slot_names = [A, Abar, R, Self] if slots is None else slots
        if slots is None and mover_type == ListMovers:
            slot_names.append(E)
        self.slots = [Slot(slot_name) for slot_name in slot_names]


    def _constant_maker(self, word, value=None, label=None, conj=False, algebra: Algebra = None):
        """
        default function for making constants.
        Given a word, use the inner algebra's make_leaf method to make the inner term, and adds empty movers.
        :return AlgebraOp with expression as function
        @param word: word as input to inner algebra's constant maker
        @param value: optional value for inner algebra object. Default will be automatically built from 'word'.
        @param conj: True if this is a conjunction
        @param label: optional label for MG operation. Default will be 'word'.
        @algebra: algebra: just for synchronous case. Which inner algebra to use.
        """
        if algebra is None:
            algebra = self.inner_algebra
        if label is None:
            label = str(word)
        expression = Expression(inner_term=algebra.make_leaf(word, function=value), movers=self.mover_type())
        expression.mg_type.conj = conj
        return AlgebraOp(label, expression)

    def ops_repr(self):
        """
        :return: string representation of the algebra operations
        """
        ret = ""
        for op in self.ops:
            ret += f"\n\t{op}"
        return ret

    def constant_maker_repr(self):
        """
        :return: string representation of the behaviour of the constant maker
        """
        if isinstance(self.inner_algebra, HMIntervalPairsAlgebra):
            w = Interval(3)
        else:
            w = "w"
        try:
            return f"  constants {w} -> {self.constant_maker(w).function}\n"
        except:
            return "constant maker exists"

    def merge1(self, inner_op: AlgebraOp, functor: Expression, other: Expression,
               prepare=None, adjoin=False, reverse=False):
        """
        MERGE1 operation (merge non-mover), applying inner_op to inner items
        and combining movers, respecting SpIC.
        Used within a lambda function as the function of a MG AlgebraOp for Merge 1.
        @param reverse: bool: if True, apply inner_op to (arg2, arg1).
                                For algebras that don't necessarily keep the head on the right.
        @param adjoin: bool: if True, allow adjunct's movers to be a subset
                            of the head's
        @param prepare: function from 2 inner terms to 2 inner terms, preparing the args for
                        merge. e.g. for head movement
        @param inner_op: AlgebraOp of inner algebra to attach functor to other with.
        @param functor: Expression (the head/selector)
        @param other: Expression (the argument)
        @return: Expression or None if operation undefined
        """
        log(f"\nMerge1 {functor} and {other}")
        log(f"\nInner op: {inner_op}")
        assert isinstance(inner_op, AlgebraOp), f"Inner op {inner_op} is of type {type(inner_op)}"

        assert isinstance(functor.inner_term, AlgebraTerm) and isinstance(other.inner_term, AlgebraTerm),\
            (f"inner terms aren't terms! They are {type(functor.inner_term).__name__}"
             f" and {type(other.inner_term).__name__}.")

        try:
            prepared_functor_inner, prepared_other_inner = self.apply_prepare(prepare, functor, other.inner_term)
            new_children = [prepared_functor_inner, prepared_other_inner]
            if reverse:
                new_children.reverse()
                logger.debug(f"reversed children: {new_children}")
            new_inner_term = AlgebraTerm(inner_op, new_children)
        except HMError as e:
            log(f"{type(e).__name__} in Merge1: {e}")
            raise e

        try:
            new_movers = functor.movers.combine_movers(other.movers, adjoin=adjoin)
        except SMCViolation as er:
            raise SMCViolation(f"{type(er).__name__} in Merge1: {er}")
        except ATBError as er:
            raise ATBError(f"{type(er).__name__} in Merge1: {er}")

        new_type = MGType(conj=functor.mg_type.conj, lexical=False)
        ret = Expression(new_inner_term, new_movers, new_type)
        log(f"output: {ret}")
        return ret

    def merge2(self, to_slot: Slot, functor: Expression, other: Expression,
               prepare=None, adjoin=False, a=False, epp=False, multiple=False):
        """
        MERGE2 operations (merge a mover)
        Applies inner_op if it's HM and stores rest, otherwise stores inner item
        Used within a lambda function as the function of a MG AlgebraOp for Merge 2.
        @param multiple:
        @param epp:
        @param a:
        @param adjoin: Bool: if true, this is an adjoin function
        @param prepare: prepare function
        @param to_slot: Slot
        @param functor: Expression
        @param other: Expression
        @return: Expression or None if operation undefined
        """
        log(f"\nMerge2 {functor} and {other}")

        # apply prepare function to inner terms
        log(f"prepare: {prepare}, to_slot: {to_slot.name}")
        new_inner_term, mover = self.apply_prepare(prepare, functor, other.inner_term)

        try:
            # combine movers and add new one
            combined_movers = functor.movers.combine_movers(other.movers,
                                                            adjoin=adjoin)
            log(f"combined movers: {combined_movers}")
            new_to_slot = deepcopy(to_slot)
            new_to_slot.add_contents(mover)
            log(f"new to_slot: {new_to_slot}")
            new_movers = combined_movers.add_mover(new_to_slot)
        except SMCViolation as er:
            log(f"{type(er).__name__} in Merge2: {er} \n {functor} \n {other}")
            raise SMCViolation(f"{type(er).__name__} in Merge2 to {to_slot.name}: {er}")
        except MGError as er:
            log(f"{type(er).__name__} in Merge2: {er} \n {functor} \n {other}")
            raise MGError(f"{type(er).__name__} in Merge2 to {to_slot.name}: {er}")
        except ATBError as er:
            raise ATBError(f"{type(er).__name__} in Merge2: {er}")

        new_type = MGType(conj=functor.mg_type.conj, lexical=False)
        new_expression = Expression(new_inner_term, new_movers, new_type)
        assert new_inner_term is not None, "Uh oh, new inner term is None!"
        log(f"output: {new_expression}")
        return new_expression

    def apply_prepare(self, prepare,
                      functor: Expression,
                      other: AlgebraTerm) -> tuple[AlgebraTerm, AlgebraTerm]:
        """
        Utility function: checks whether prepare can be applied, and if so, applies it to inner terms.
        @param prepare: tree homomorphism from a PreparePackages object
        @param functor: Expression.
        @param other: inner AlgebraTerm.
        @return: updated functor inner term, updated other inner term. (AlgebraTerms)
        """
        if prepare is not None:
            assert prepare_allowed(functor,
                                   prepare.__name__
                                   ), f"{prepare.name} not allowed with functor of type {functor.mg_type}"
            if prepare.__name__ != "simple":
                logger.debug(f"Applying prepare: {prepare.__name__}")
            return prepare(functor.inner_term, other)
        else:
            return functor.inner_term, other

    def move1(self, from_slot: str, e: Expression, inner_op: AlgebraOp or None = None, index=0, prepare=None,
              reverse=False):
        """
        MOVE1 (final move) operations.
        Take mover out of to_slot, concatenate left or right.
        Used within a lambda function as the function of a MG AlgebraOp for Move 1.
        @param prepare: prepare function. Note that this might be impossible in anything resembling a standard MG.
        @param index: int: for ListMovers, which mover to take from a multiple slot.
        @param from_slot: str: name of slot to get the mover from.
        @param inner_op: AlgebraOp of inner algebra to attach mover to main term with.
        @param reverse: bool: if True, apply inner_op to (arg2, arg1).
                        For algebras that don't necessarily keep the head on the right.
        @param e: Expression: input.
        @return: Expression: output.
        """

        log(f"\nMove1 {e}")
        log(f"inner op: {inner_op} from_slot: {from_slot}:{index}")
        if inner_op is None:
            inner_op = self.inner_algebra.get_op_by_name("concat_left")

        try:
            # remove mover from movers
            if self.mover_type == ListMovers:
                mover_contents, new_movers = e.movers.remove_mover(from_slot, index)
            else:
                mover, new_movers = e.movers.remove_mover(from_slot)
                mover_contents = mover.contents
        except ExistenceError as err:
            raise MGError(f"{type(err).__name__} in Move1: {err}")
        except Exception as err:
            raise MGError(f"Weird! {type(err).__name__} in Move1: {err}")
        try:
            log(f"move1 of {e.inner_term} and {mover_contents}")
            prepared_functor_inner, prepared_mover = self.apply_prepare(prepare, e, mover_contents)
            new_children = [prepared_functor_inner, prepared_mover]
            if reverse:
                new_children.reverse()
                logger.debug(f"reversed children: {new_children}")
            new_inner_term = AlgebraTerm(inner_op, new_children)
        except HMError as er:
            log(f"{type(e).__name__} in Move1: {er}\n {e}")
            raise MGError(f"{type(er).__name__} in Move1 from {from_slot}: {er}")

        new_type = MGType(conj=e.mg_type.conj, lexical=False)
        new_expression = Expression(new_inner_term, new_movers, new_type)
        log(f"output: {new_expression}")
        return new_expression

    def move2(self, from_slot: str, to_slot: Slot, expression: Expression, index=0, prepare=None):
        """
        MOVE2 (non-final move) from one mover to_slot to another.
        Used within a lambda function as the function of a MG AlgebraOp for Move 2.
        @param index: index if using ListMovers.
        @param prepare: prepare function (e.g. add a trace).
        @param from_slot: str: name of the to_slot we take mover from.
        @param to_slot: Slot: empty Slot to put mover into.
        @param expression: Expression.
        @return: Expression.
        """
        log(f"\nMove2 of {expression}")
        log(f"from_slot: {from_slot}:{index}, to to_slot: {to_slot.name}")
        # remove mover from movers
        new_to_slot = deepcopy(to_slot)
        if self.mover_type == ListMovers:
            mover_contents, new_movers = expression.movers.remove_mover(from_slot, index)
        else:
            mover, new_movers = expression.movers.remove_mover(from_slot)
            mover_contents = mover.contents
        try:
            log(f"new to_slot: {to_slot}")
            prepared_functor_inner, prepared_mover = self.apply_prepare(prepare, expression, mover_contents)
            new_to_slot.add_contents(prepared_mover)
            new_movers = new_movers.add_mover(new_to_slot)
        except SMCViolation as er:
            log(f"SMCViolation in Move2: {er}\n {expression}")
            raise SMCViolation(f"{type(er).__name__} in Move2: {er}")
        except MGError as er:
            raise MGError(f"{type(er).__name__} in Move2: {er}")
        except ATBError as er:
            raise ATBError(f"{type(er).__name__} in Move2: {er}")

        log(f"output: {Expression(prepared_functor_inner, new_movers)}")

        new_type = MGType(conj=expression.mg_type.conj, lexical=False)
        new_expression = Expression(expression.inner_term, new_movers, new_type)
        log(f"output: {new_expression}")
        return new_expression


def leaves_in_order(term):
    """
    Given a term over a MinimalistAlgebra, returns the string yield of the term qua tree;
        i.e. the names of the leaves
    @param term: AlgebraTerm over a MinimalistAlgebra
    @return: string list
    """

    def leaves_in_order_inner(term: AlgebraTerm, leaves_to_add_to):
        if term.children is None:
            leaves_to_add_to.append(term.parent.name)
        else:
            for child in term.children:
                leaves_in_order_inner(child, leaves_to_add_to)

    leaves = []
    leaves_in_order_inner(term, leaves)
    return leaves


def leaf_spellouts_in_order(term):
    """
    Given a term over a MinimalistAlgebra, returns the spellouts of the leaves in order
    in a tuple so it's hashable
    @param term: AlgebraTerm over a MinimalistAlgebra
    @return: inner algebra object list
    """

    def leaves_in_order_inner(term: AlgebraTerm, leaves_to_add_to):
        if term.children is None:
            leaves_to_add_to.append(term.evaluate().spellout())
        else:
            for child in term.children:
                leaves_in_order_inner(child, leaves_to_add_to)

    leaves = []
    leaves_in_order_inner(term, leaves)
    return leaves


if __name__ == "__main__":
    # from minimalist_parser.minimalism.mgbank_input_codec import read_tsv_corpus_file
    # from minimalist_parser.minimalism.mgbank_input_codec import tree2term
    # from prepare_packages.triple_prepare_package import HMTriplesPreparePackages
    #
    # corpus = read_tsv_corpus_file("../../data/processed/seq2seq/sampled/to_predict.tsv")
    #
    # nltk_tree = corpus[0][1]
    # tree = Tree.nltk_tree2tree(nltk_tree)
    #
    #
    # t = tree2term(tree, HMTripleAlgebra(), HMTriplesPreparePackages(), ListMovers)
    #
    # print(leaves_in_order(t))
    # nltk_tree.draw()

    from minimalist_parser.algebras.am_algebra_untyped import AMAlgebra
    from minimalist_parser.algebras.algebra_objects.graphs import SGraph

    like = SGraph(
        {0, 1, 2},
        {
            1: [(0, "ARG0"), (2, "ARG1")]
        },
        {1: "like"},
        {"S": 0, "O": 2},
        1
    )

    tried = SGraph(
        {0, 1, 2},
        {
            1: [(0, "ARG0"), (2, "ARG1")]
        },
        {1: "try"},
        {"S": 0, "O": 2},
        1
    )

    mary = SGraph(
        {0, 1},
        edges={0: [(1, "name")]},
        node_labels={0: "person", 1: "Mary"},
        root=0
    )

    cat = SGraph(
        {0},
        node_labels={0: "cat"},
        root=0
    )

    whistling = SGraph(
        {0, 1, 2},
        {
            1: [(2, "ARG0")],
            0: [(1, "mod")]
        },
        {1: "whistle"},
        {"S": 2, "M": 0},
        1
    )

    m = SGraph(
        {0, 1, 2},
        {1: [(2, "with"), (0, "mod")]},
        {0: "cute", 2: "glee"},
        sources={"M": 1},
        root=0
    )

    vocabulary = {"mary": mary,
                  "tried": tried,
                  "like": like,
                  "cute": m,
                  "whistling": whistling
                  }

    am_alg = AMAlgebra()
    am_alg.add_constants({name: AlgebraOp(name, vocabulary[name]) for name in vocabulary},
                         default=am_alg.default_constant_maker)
    # am_alg.add_constants({name: vocabulary[name] for name in vocabulary}, default=am_alg.default_constant_maker)

    print("VOCAB:")
    for constant in am_alg.constants:
        print(constant, am_alg.constants[constant])

    mg = MinimalistAlgebra(am_alg)
    for inner_operation in mg.inner_algebra.ops:
        op = MinimalistFunction(mg.merge1, inner_op=mg.inner_algebra.ops[inner_operation])
        mg.add_op(op)

    print(mg)

    # e1 = Expression(inner_term=am_alg.make_leaf("like"), movers=Movers())
    # e2 = Expression(inner_term=am_alg.make_leaf("cat"), movers=Movers())
    # e = merge1(am_alg.ops["App_S"], e1, e2)
    # print(e)
    # print(e.inner_term.evaluate())

    mg_app_s = MinimalistFunction(mg.merge1, inner_op=am_alg.ops["App_S"])
    # t = AlgebraTerm(mg_app_s, [AlgebraTerm(AlgebraOp("like", e1)), AlgebraTerm(AlgebraOp("cat", e2))])
    t = AlgebraTerm(mg.ops["merge1+App_S"], [mg.make_leaf("like"), mg.make_leaf("cat")])

    print(t)
    print(t.evaluate().spellout())

    ...

