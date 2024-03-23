"""
Minimalist Algebra in which inner items are lists of inner algebra terms from a list of algebras
"""
import os.path
import sys
# from .. import logging, root_logging
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any

from overrides import overrides

from minimalist_parser.algebras.algebra_objects.intervals import Interval, PairItem
from minimalist_parser.algebras.hm_algebra import HMAlgebra
from minimalist_parser.algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from minimalist_parser.minimalism.mg_types import MGType
from minimalist_parser.minimalism.prepare_packages.addressed_triple_prepare_package import \
    HMAddressedTriplesPreparePackages
from minimalist_parser.minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare
from minimalist_parser.minimalism.prepare_packages.prepare_packages_bare_trees import PreparePackagesBareTrees
from minimalist_parser.minimalism.prepare_packages.prepare_packages_hm import PreparePackagesHM
from .minimalist_algebra import Expression, prepare_allowed, MinimalistFunction, MinimalistAlgebra
from ..algebras.algebra import AlgebraTerm, Algebra, AlgebraError, AlgebraOp
from ..minimalism.mg_errors import *
from ..minimalism.movers import Movers, Slot, DSMCMovers, DSMCAddressedMovers

from ..minimalism.prepare_packages.prepare_packages import PreparePackages
from ..trees.trees import Tree

import logging
VERBOSE = True
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug
from ..algebras.string_algebra import BareTreeStringAlgebra


# VERBOSE = False


# def log(string):
#     if VERBOSE:
#         print(string)


class InnerAlgebraInstructions:
    def __init__(self, op_name: str = None, prepare: str = None, leaf_object=None, reverse=False, algebra_op=None):
        """
        Stores the necessary components for creating a MinimalistFunction for the particular inner algebra.
        @param op_name: the name of the inner op (to look it up in alg.ops or to give to alg.constant_maker.
        @param prepare: the name of the prepare operation, if any, for Merge1 or Move1.
        @param leaf_object: If this is a leaf, you can explicitly give the algebra object,
                            which will be used in place of whatever the default constant maker will create.
        @param reverse: For Merge1 or Move1, if True, the operands to the inner operation will be reversed.
                        Used for inner algebras that don't always keep the head on the left.
                        (Minimalist Algebras as defined here always do.)
        @param algebra_op: if the algebra operation is given directly, we use this as the inner op, or leaf operation.
        """
        self.op_name = op_name
        self.prepare = prepare
        self.leaf_object = leaf_object
        self.reverse = reverse
        self.algebra_op = algebra_op

    def __repr__(self):
        ret = "{"
        if self.op_name is not None:
            ret += self.op_name
            ret += ", "
        if self.prepare is not None:
            ret += self.prepare
            ret += ", "
        if self.leaf_object is not None:
            ret += str(self.leaf_object)
            ret += ", "
        if self.reverse:
            ret += "reversed"
            ret += ", "
        if self.algebra_op is not None:
            ret += self.algebra_op.name
        return ret + "}"

    def __eq__(self, other):
        logger.debug(f"Comparing {self} and {other}: {repr(self) == repr(other)}")
        return repr(self) == repr(other)


class MinimalistAlgebraSynchronous(MinimalistAlgebra):
    """
    A synchronous Minimalist Algebra: i.e. this has multiple Inner Algebras for structure-building.
    Attributes:
        inner_algebras: dict {Algebra: PreparePackages}
        inner_algebra: for compatability with parent class,
            the first algebra in the input parameter inner_algebras is stored here as a default inner algebra.
    """
    def __init__(self, inner_algebras: list[Algebra], mover_type=Movers, prepare_packages=None, slots=None):
        """
        :param inner_algebras: list of algebras forming the inner algebras of the interpretations
        :param mover_type: Type of mover storage (default parent class Movers)
        :param prepare_packages: list of PreparePackages objects, one for each algebra
        :param slots: list of Slot objects, default [A, Abar, R, Self] (plus E if ListMovers)
        """
        if len(inner_algebras) == 0:
            raise ValueError("At least one Inner Algebra is required")
        # make the first inner algebra on the list the default algebra (inner_algebra)
        # so that this is a functional minimalist algebra
        super().__init__(mover_type=mover_type, prepare_packages=prepare_packages, slots=slots,
                         inner_algebra=inner_algebras[0])
        self.inner_algebras = {}
        # make a default inner algebra for compatability with MGs with only one, unspecified, inner algebra
        if prepare_packages is None:
            prepare_packages = [None] * len(inner_algebras)
        for algebra, prepare in zip(inner_algebras, prepare_packages):
            self.inner_algebras[algebra] = prepare

    def __repr__(self):
        s = super().__repr__()
        s += "\nInner Algebras:"
        for algebra in self.inner_algebras:
            s += f"\n{algebra}: {self.inner_algebras[algebra]}"
        return s

    def add_default_operations(self):
        """
        Only if all inner algebras are HM algebras, which come with the appropriate operation names.
        Adds merge/move right and left, merge2/move1 for every mover slot, and move2 for every pair of mover slots.
        """
        if all([isinstance(a, HMAlgebra) for a in self.inner_algebras]):

            m = MinimalistFunctionSynchronous(self, self.merge1,
                                              {a: InnerAlgebraInstructions("concat_right") for a in self.inner_algebras},
                                              name="Merge1_right")
            self.ops[m.name] = m
            m = MinimalistFunctionSynchronous(self, self.merge1,
                                              {a: InnerAlgebraInstructions("concat_left", reverse=isinstance(a, BareTreeStringAlgebra)) for a in self.inner_algebras},
                                              name="Merge1_left")
            self.ops[m.name] = m
            for slot in self.slots:
                m = MinimalistFunctionSynchronous(self, self.merge2, to_slot=slot, name=f"Merge2_{slot}")
                self.ops[m.name] = m
                m = MinimalistFunctionSynchronous(self, self.move1,
                                                  {a: InnerAlgebraInstructions("concat_right") for a in self.inner_algebras},
                                                  from_slot=slot,
                                                  name=f"Move1_right_{slot}")
                self.ops[m.name] = m
                m = MinimalistFunctionSynchronous(self, self.move1,
                                                  {a: InnerAlgebraInstructions("concat_left", reverse=isinstance(a, BareTreeStringAlgebra)) for a in self.inner_algebras},
                                                  from_slot=slot.name,
                                                  name=f"Move1_left_{slot}")
                self.ops[m.name] = m
                for to_slot in self.slots:
                    m = MinimalistFunctionSynchronous(self, self.move2, from_slot=slot.name, to_slot=to_slot,
                                                      name=f"Move2_{slot.name}_{to_slot}")
                    self.ops[m.name] = m
        else:
            logger.warning(f"unable to make default operations, "
                           f"since at least one inner algebra lacks concat_right or concat_left operation names: {e}")

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
        inner_ops = {algebra: InnerAlgebraInstructions(word, leaf_object=value)}
        return MinimalistFunctionSynchronous(self, inner_ops=inner_ops, name=label, conj=conj)

    def synchronous_term_from_minimalist_term(self, input_term: AlgebraTerm, inner_algebra: Algebra):
        if input_term.is_leaf():
            op = MinimalistFunctionSynchronous(minimalist_algebra=self,
                                               inner_ops={
                                                   inner_algebra: (input_term.parent.name, None)},
                                               name=input_term.parent.name
                                               )  # function.inner_term.parent.
            return SynchronousTerm(op)
        else:
            op = self.synchronous_function_from_minimalist_function(input_term.parent, inner_algebra)

            return SynchronousTerm(op,
                                   [self.synchronous_term_from_minimalist_term(kid, inner_algebra)
                                    for kid in input_term.children])

    def synchronous_function_from_minimalist_function(self, input_function: MinimalistFunction, inner_algebra: Algebra):
        return MinimalistFunctionSynchronous(self,
                                             minimalist_function=input_function.mg_function_type,
                                             inner_ops={inner_algebra: (input_function.inner_op.__name__,
                                                                        input_function.prepare)},
                                             from_slot=input_function.from_slot,
                                             to_slot=input_function.to_slot,
                                             index=input_function.index,
                                             adjoin=input_function.adjoin,

                                             )

    def make_leaf(self, name, inner_instructions: InnerAlgebraInstructions or None = None, conj=False):
        return SynchronousTerm(MinimalistFunctionSynchronous(minimalist_algebra=self,
                                                             inner_ops=inner_instructions,
                                                             name=name,
                                                             conj=conj))


class MinimalistFunctionSynchronous(AlgebraOp):
    def __init__(self, minimalist_algebra: MinimalistAlgebraSynchronous, minimalist_function=None,
                 inner_ops: dict[Algebra: InnerAlgebraInstructions] or None = None, from_slot=None, to_slot=None,
                 index=None, adjoin=False, name=None, conj=False):
        """
        :param minimalist_function: merge1 or whatever (default None).
        :param minimalist_algebra: MinimalistAlgebraSynchronous.
        :param inner_ops: dict from inner alg name to InnerAlgebraInstructions.
        :param from_slot: for Move; Slot from which to extract the mover
        :param to_slot: for Merge 2 and Move 2: Slot in which to store the mover
        :param index: for ListMovers, Move: index from which to take mover
        :param adjoin: Boolean; whether this is an adjoin operation
        :param name: usually built up out of parts, but can be given here
        """
        self.minimalist_function = minimalist_function
        self.minimalist_algebra = minimalist_algebra
        self.from_slot = from_slot
        self.to_slot = to_slot
        self.index = index
        self.adjoin = adjoin
        self.inner_ops = inner_ops
        self.conj = conj  # for lexical items

        if name is None:
            # build operation name
            name = ""
            if minimalist_function:
                name += minimalist_function.__name__
            if from_slot:
                name += f"+{from_slot}"
            if index is not None:
                name += f".{index}"
            if to_slot:
                name += f"+{to_slot.name}"
            if self.inner_ops is not None:
                for alg in self.inner_ops:
                    info = self.inner_ops[alg]
                    if info.op_name is not None:
                        name += f"+{info.op_name}"
                    if info.prepare not in [None, "simple"]:
                        name += f"+{info.prepare}"
            if self.adjoin:
                name += "+adjoin"
        self.name = name

        def apply_all(children):
            """
            Applies MinimalistFunction equivalent for each inner algebra to children
            :param children: this needs to be a function from a list of children
            :return: list of output Expressions for each algebra
            """
            return [self.single_function(a).function(children) for a in self.inner_ops]

        # the AlgebraOp has apply_all as its function
        super().__init__(name, apply_all)

    def __repr__(self):
        return self.name

    def single_function(self, inner_algebra: Algebra):
        """
        Generate a single minimalist function with the chosen inner algebra
        :param inner_algebra: The inner algebra to be used
        :return: A MinimalistFunction
        """
        # Inner op
        if self.inner_ops is None:
            inner = None
            prepare = None
            reverse = False
        else:
            log(f"looking in inner ops: {self.inner_ops}")
            instructions = self.inner_ops[inner_algebra]
            logger.debug(f"{inner_algebra}: {instructions}")
            if not isinstance(instructions, InnerAlgebraInstructions):
                logger.warning(f"should be instructions")

            # get the inner operation
            if instructions.algebra_op is not None:
                # we've been given an explicit AlegraOp, so just use it.
                inner = instructions.algebra_op
            else:
                inner_name = instructions.op_name
                if inner_name is not None:
                    inner = inner_algebra.get_op_by_name(inner_name)
                    assert inner is not None, "somehow inner is none even though we found it."
                else:
                    inner = None

            # Prepare
            prepare_name = instructions.prepare
            if prepare_name is not None:
                assert isinstance(prepare_name, str), "prepare should just be a name"
                if not inner_algebra in self.minimalist_algebra.inner_algebras:
                    raise ValueError(f"{inner_algebra} not among inner algebras {self.minimalist_algebra.inner_algebras.keys()}")
                prepare_packages = self.minimalist_algebra.inner_algebras[inner_algebra]
                if not isinstance(prepare_packages, PreparePackages):
                    raise TypeError(f"{prepare_packages} should be PreparePackages but is {type(prepare_packages)}")
                prepare = getattr(self.minimalist_algebra.inner_algebras[inner_algebra], prepare_name)
            else:
                prepare = None

            reverse = instructions.reverse

        if inner is None and self.name in ["move1", "merge1"]:
            logger.warning(f"no inner op for {self.name}")

        try:
            return MinimalistFunction(self.minimalist_function,
                                      name=self.name,
                                      inner_op=inner,
                                      from_slot=self.from_slot,
                                      to_slot=self.to_slot, index=self.index, adjoin=self.adjoin,
                                      prepare=prepare,
                                      reverse=reverse)

        except MGError as e:
            logger.warning(f"Didn't make a minimalist op from {self} into {inner_algebra}: {e}")


class SynchronousTerm(AlgebraTerm):
    """
    SynchronousTerm is a minimalist AlgebraTerm that has interpretations into multiple inner algebras.
    The minimalist algebra it is defined over is a MinimalistAlgebraSynchronous.
    The AlgebraOps are of type MinimalistFunctionSynchronous.
    The objects of the algebra are lists of expressions,
        which is probably a little silly, since the expressions have the same structure,
        but it makes it straightforward to grab a single interpretation over a single inner algebra.
    The interp(inner_algebra) function is like evaluate but for a single inner algebra.
    """

    def __init__(self, parent: MinimalistFunctionSynchronous,
                 children: list[Tree] or None = None):
        super().__init__(parent, children=children)
        # self.minimalist_algebra = minimalist_algebra

    def spellout(self, algebra=None):
        """
        If possible, spell out with the given or default inner algebra
        """
        if algebra is None:
            algebra = self.parent.minimalist_algebra.inner_algebra
        return self.interp(algebra).spellout()

    def to_minimalist_term(self, algebra=None):
        """
        Get a term over a regular MinimalistAlgebra for the given, or default, inner algebra.
        :param algebra: The inner algebra
        :return: AlegebraTerm over MinimalistAlgebra with inner algebra 'algebra'.
        """
        if algebra is None:
            algebra = self.parent.minimalist_algebra.inner_algebra
        if self.is_leaf():
            e = self.interp(algebra)
            return AlgebraTerm(AlgebraOp(name=self.parent.name, function=e))
        else:
            function = self.parent.single_function(algebra)
            log(f"inner_op: {function.inner_operation}")
            return AlgebraTerm(function, [child.to_minimalist_term(algebra) for child in self.children])

    def evaluate(self):
        return self.interp()

    def view_inner_term(self, algebra):
        """
        Interpret into the given algebra and visualise the inner term with NLTK.
        """
        expr = self.interp(algebra)
        expr.inner_term.to_nltk_tree().draw()

    def interp(self, algebra=None) -> Expression:
        """
        Recursively evaluate the tree,
            making it into an algebra term over a Minimalist Algebra over the given inner algebra
            and evaluating it to an Expression of the form:
            (term over inner algebra, movers).
        Skips any operations with children that evaluate to None.
            This is so that e.g. an AM algebra can skip over function words
        :param algebra: the inner algebra for the minimalist algebra to use.
                        Default: the inner algebra stored in parent's
                            Synchronous Minimalist Algebra's attribute inner_algebra,
                            which is by default the first in the list
        """
        # Get a default inner algebra if necessary
        if algebra is None:
            logger.debug("using default algebra")
            inner_algebra = self.parent.minimalist_algebra.inner_algebra
        else:
            inner_algebra = algebra
            logger.debug(f"using given algebra, {inner_algebra.name}")

        # get mover type
        if inner_algebra.meta.get("addresses", False):
            mover_type = DSMCAddressedMovers
        else:
            mover_type = self.parent.minimalist_algebra.mover_type

        log(f"\nInterpreting term into {inner_algebra.name}: {self}")
        assert issubclass(type(self.parent),
                          MinimalistFunctionSynchronous), (f"Should be a MinimalistFunctionSynchronous, "
                                                           f"but instead is a {type(self.parent).__name__}")
        if self.is_leaf():
            log(f"\ninterpreting leaf: {self.parent} of type {type(self.parent).__name__}")
            log(f"\n{self.parent.inner_ops}")

            # Look the constant name up in the inner_ops dictionary from alg name to (op name, prepare)
            # these are constants, so op name is going to be fed to the constant maker.
            if self.parent.inner_ops is not None and inner_algebra in self.parent.inner_ops:
                if self.parent.inner_ops[inner_algebra] is not None:
                    log("using given constant name")
                    instructions = self.parent.inner_ops[inner_algebra]
                    if instructions.algebra_op is not None:
                        # use the AlgebraOp we've been given
                        inner_term = AlgebraTerm(instructions.algebra_op)
                    else:
                        inner_op_name = instructions.op_name
                        if inner_op_name is None:
                            log(f"{self} has no interpretation in {inner_algebra}")
                            return
                        else:
                            assert isinstance(inner_op_name,
                                              str), f"Inner op name {inner_op_name} should be a string but is a {type(inner_op_name)}"
                            inner_term = inner_algebra.make_leaf(inner_op_name, function=instructions.leaf_object)
                else:
                    log(f"{self} has no interpretation in {inner_algebra}")
                    return
            else:
                log("using MG leaf label")
                if self.parent.inner_ops is None:
                    log("inner ops is None")
                else:
                    log(f"{inner_algebra} not found in {self.parent.inner_ops}")
                word = self.parent.name
                try:
                    inner_op = inner_algebra.constant_maker(word=word, label=self.parent.name)
                except TypeError:
                    inner_op = inner_algebra.constant_maker(word)

                inner_term = AlgebraTerm(inner_op)

            # log(f"made inner leaf term which evaluates to {inner_term.evaluate()}")
            return Expression(inner_term, mover_type(),
                              mg_type=MGType(lexical=True, conj=self.parent.conj))

        # internal node
        else:
            try:
                logger.debug(f"\nevaluating {self.parent} of type {type(self.parent)} with {len(self.children)} children")
                # logger.debug(f"children have types {[type(kid).__name__ for kid in self.children]}")
            except TypeError as e:
                logger.warning(e)
                logger.warning(f"\n***********parent: {self.parent.name}")
                logger.warning(type(self.parent))
                logger.warning(f"function {self.parent.function}")
                logger.warning(f"name {self.parent.name}")
                logger.warning(f"***************\n")
            logger.debug(f"Applying {self.parent} to {self.children}"
                         f" of types {[type(kid).__name__ for kid in self.children]}")
            if self.parent.function is None:
                raise AlgebraError(f"function of {self.parent} is None")
            if not callable(self.parent.function):
                raise AlgebraError(f"Internal node interpretation of type"
                                   f" {type(self.parent.function).__name__}"
                                   f" is not a callable function")
            try:
                # recursive step
                interpreted_children = [kid.interp(inner_algebra) for kid in self.children]
                # If any children are None, skip this function for this algebra
                if None in interpreted_children:
                    try:
                        real_kid = [kid for kid in interpreted_children if kid is not None][0]
                        log(f"Skipping {self.parent} due to meaningless child among {interpreted_children}")
                        output = real_kid
                    except IndexError:
                        raise MGError(f"both children of {self.parent.name} are None")
                else:
                    mg_op = self.parent.single_function(inner_algebra)
                    log(f"created MG function: {mg_op}")
                    output = mg_op.function(interpreted_children)
                logger.debug(f"building {output} of type {type(output).__name__}")
                return output
            except IndexError:
                raise AlgebraError(
                    f"{self.parent} requires more arguments than provided ({[kid.parent for kid in self.children]})")
            except TypeError:
                logger.error(f"parent function {self.parent.function}")
                raise

    def add_algebra(self, algebra: Algebra, other_term: AlgebraTerm):
        """
        Add a new interpretation over a different inner algebra, creating a new term
        :param algebra: inner Algebra being added
        :param other_term: AlgebraTerm over MinimalistAlgebra with different inner algebra
        :return: new SynchronousTerm with added inner algebra
        """
        return self._add_algebra(algebra, other_term)[0]

    def _add_algebra(self, algebra: Algebra, other_term: AlgebraTerm):
        """
        Add a new interpretation over a different inner algebra, creating a new term.
        Use the unprotected version to just get back the new SynchronousTerm.
        :param algebra: inner Algebra being added
        :param other_term: AlgebraTerm over MinimalistAlgebra with inner algebra = algebra
        :return: updated SynchronousTerm subterm, relevant subterm of other term over MinimalistAlgebra
        """
        new_term = copy(self)
        if new_term.is_leaf():
            # extract the inner algebra's AlgebraOp leaf
            new_term.parent.inner_ops[algebra] = InnerAlgebraInstructions(
                other_term.parent.function.inner_term.parent.name)
            return new_term, other_term
        else:
            # extract the inner op name and inner prepare function
            # something's weird -- sometimes we have a name and sometimes a __name__
            name = None
            try:
                name = other_term.parent.inner_operation.__name__
            except AttributeError:
                logger.warning(
                    f"{other_term.parent.inner_operation} should perhaps have a '__name__' attribute but doesn't.")
            try:
                # something's weird -- sometimes we have a name and sometimes a __name__
                name = other_term.parent.inner_operation.name
            except AttributeError:
                logger.warning(
                    f"{other_term.parent.inner_operation} should perhaps have a 'name' attribute but doesn't.")
            if name is None:
                raise AttributeError(f"{other_term.parent.inner_operation} has neither name nor __name__")
            new_term.parent.inner_ops[algebra] = InnerAlgebraInstructions(name, other_term.parent.prepare,
                                                                          reverse=other_term.parent.reverse)
            # recurse
            new_pairs = [kid._add_algebra(algebra, other_kid) for (kid, other_kid) in
                         zip(new_term.children, other_term.children)]
            # we just want the new_term, not the subterm of the other term
            new_term.children = [pair[0] for pair in new_pairs]
            # return the current updated subterm and other subterm
            return new_term, other_term





if __name__ == "__main__":
    from ..algebras.string_algebra import BareTreeStringAlgebra
    from ..algebras.am_algebra_untyped import am_alg
    from minimalist_parser.algebras.algebra_objects.graphs import SGraph
    from minimalist_parser.algebras.algebra_objects.triples import Triple
    from ..algebras.hm_triple_algebra import HMTripleAlgebra
    from .prepare_packages.triple_prepare_package import HMTriplesPreparePackages
    from ..convert_mgbank.slots import Abar, A, R, Self, E
    from minimalist_parser.algebras.algebra_objects.amr_s_graphs import AMRSGraph
    from ..algebras.am_algebra_untyped import make_predicate

    # String algebra with Bare Tree terms
    string_alg = BareTreeStringAlgebra(name="string_algebra")
    string_alg.add_constant_maker()

    # initialise triple algebra
    triple_alg = HMTripleAlgebra(name="string triples")
    triple_alg.add_constant_maker()
    triple_prepare_packages = HMTriplesPreparePackages(inner_algebra=triple_alg)
    print("***************** EMPTY ********************")
    print(triple_alg.empty_leaf_operation.function)
    print("done")
    # addresses_alg = HMTripleAlgebra(name="address triples", component_type=list)

    # initialise MG and Sync MG over triples only
    # mg = MinimalistAlgebra(triple_alg, prepare_packages=triple_prepare_packages)
    # smg = MinimalistAlgebraSynchronous([triple_alg],
    #                                    prepare_packages=[triple_prepare_packages],
    #                                    mover_type=DSMCMovers)

    # # add some consonants
    # cat = mg.make_leaf("cat")
    # # print("###################### Trying to make constants")
    # # print(cat)
    # # print(cat.evaluate())
    # # print(cat.evaluate().spellout())
    # slept = mg.make_leaf("slept")
    # # print(slept.evaluate())
    # # print(slept.evaluate().inner_term)
    # # print(slept.evaluate().inner_term.evaluate())
    # # print("############# DONE ##############")
    # merge = MinimalistFunction(mg.merge1, "merge", inner_op=triple_alg.concat_right)
    # cat_slept = AlgebraTerm(merge, [slept, cat])
    # # print(cat_slept.evaluate())
    #
    # # turn term into a SynchronousTerm
    # tree_sync = smg.synchronous_term_from_minimalist_term(cat_slept, inner_algebra=triple_alg)
    # # print(tree_sync)
    # print("################# Interp")
    # print(tree_sync.interp(triple_alg))
    # print("DONE\n")
    # AM algebra update
    and_g = AMRSGraph({0, 1, 2}, edges={0: [(1, "op1"), (2, "op2")]}, root=0, sources={"OP1": 1, "OP2": 2},
                      node_labels={0: "and"})
    and_g_op = AlgebraOp("and_s", and_g)

    dream_g = make_predicate(label="dream-01", arg_numbers=[0])
    am_alg.add_constants({"and_s": and_g_op,
                          "dreamt": AlgebraOp("dreamt", dream_g)}, default=am_alg.default_constant_maker)

    #
    # mg_am = MinimalistAlgebra(am_alg)
    # print(am_alg.constants)
    # cat_g = mg_am.make_leaf("cat")
    # slept_g = mg_am.make_leaf("slept_mg_name")
    # print(slept_g.parent.function.inner_term.evaluate())
    # merge_app_s = MinimalistFunction(mg.merge1, "merge", inner_op=am_alg.ops["App_S"])
    #
    # cat_slept_g = AlgebraTerm(merge_app_s, [slept_g, cat_g])
    # print(cat_slept_g.evaluate().spellout())
    #
    # tree_sync.add_algebra(am_alg, cat_slept_g)
    # print("***spellout\n", tree_sync.interp(am_alg).spellout())
    # print("***spellout\n", tree_sync.interp(triple_alg).spellout())
    #
    # print("\n*** testing evaluate")
    # print(tree_sync.evaluate().spellout())
    #
    # print("#############")
    # print(am_alg.constants["cat_mg_name"].function)
    # print(am_alg.constants["slept_mg_name"].function)
    # print("###########")

    # example with 4 interpretations
    mg = MinimalistAlgebraSynchronous([string_alg, am_alg, triple_alg],
                                      prepare_packages=[PreparePackagesBareTrees(name="for strings",
                                                                                 inner_algebra=string_alg),
                                                        None, triple_prepare_packages], mover_type=DSMCMovers)
    # print(mg)
    #
    print("AM constants")
    for c in am_alg.constants:
        print(c)
        assert isinstance(am_alg.constants[c].function,
                          SGraph), f"{c} is not an SGraph but a {type(am_alg.constants[c].function)}"

    the = mg.constant_maker("the", string_alg)
    # print(type(the.function.inner_term.evaluate()))
    cat_s = mg.constant_maker("cat", string_alg)
    cat_g = mg.constant_maker("cat", am_alg)
    # print(cat_s.function)
    # print(cat_g.function.inner_term.evaluate())
    # print(am_alg.constants["mary"].function)

    # optionally, we give labels for the default constant maker to use if words aren't found in the constant dict
    # note that we look up the words using the name of the MinimalistFunction, below
    cat_functions = {
        am_alg: InnerAlgebraInstructions('poes'),
    }

    to_functions = {am_alg: None,  # this means there's no interpretation, so we'll skip it
                    string_alg: InnerAlgebraInstructions("to_control", leaf_object="to"),
                    triple_alg: InnerAlgebraInstructions("to_control", leaf_object=Triple("to")),
                    }

    past_functions = {string_alg: InnerAlgebraInstructions('[past]', leaf_object=string_alg.empty_leaf_operation.function),
                      triple_alg: InnerAlgebraInstructions('[past]', leaf_object=triple_alg.empty_leaf_operation.function),
                      am_alg: None,  # this means there's no interpretation, so we'll skip it
                      }

    q_functions = {string_alg: InnerAlgebraInstructions('[Q]', leaf_object=str()),
                   triple_alg: InnerAlgebraInstructions('[Q]', leaf_object=Triple()),
                   am_alg: None,  # this means there's no interpretation, so we'll skip it
                   }

    did_functions = {
        am_alg: None,  # this means there's no interpretation, so we'll skip it
    }

    # should be +conj
    and_functions = {am_alg: InnerAlgebraInstructions("and_s")}



    # Leaf algebra ops
    # cat_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=cat_functions, name='cat')
    # slept_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=slept_functions, name='slept_mg_name')
    # dreamt_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, name='dreamt')
    # tried_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, name="tried")
    # to_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=to_functions, name='to_control')
    # and_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=and_functions, name='and_s', conj=True)
    # past_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=past_functions, name='[past]')
    # did_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=did_functions, name='did')
    # q_f = MinimalistFunctionSynchronous(minimalist_algebra=mg, inner_ops=q_functions, name='[Q]')
    #
    # # turned into terms
    # # cat = SynchronousTerm(cat_f)
    # slept = SynchronousTerm(slept_f)
    # dreamt = SynchronousTerm(dreamt_f)
    # tried = SynchronousTerm(tried_f)
    # to = SynchronousTerm(to_f)
    # and_term = SynchronousTerm(and_f)
    # past = SynchronousTerm(past_f)
    # did = SynchronousTerm(did_f)
    # q = SynchronousTerm(q_f)
    #
    # leaves
    cat = mg.make_leaf("cat", cat_functions)
    slept = mg.make_leaf("slept")
    dreamt = mg.make_leaf("dreamt")
    tried = mg.make_leaf("tried")
    to = mg.make_leaf("to_control", to_functions)
    and_term = mg.make_leaf("and", and_functions, conj=True)
    past = mg.make_leaf("[past]", past_functions)
    did = mg.make_leaf("did", did_functions)
    q = mg.make_leaf("[Q]", q_functions)
    sleep = mg.make_leaf("sleep")
    dream = mg.make_leaf("dream")
    the = mg.make_leaf("the", {am_alg: None})
    dog = mg.make_leaf("dog")

    #
    # print("############### DID ###################")
    # print(did_f)
    # print(did_f.inner_ops[triple_alg])

    # print("\n\n########### AND Term ################")
    # print(and_term.evaluate().mg_type.conj)
    # print(and_term.interp(am_alg))
    # print(and_term.interp(am_alg).spellout())

    # print("t", t)
    #
    # t_string_alg = t.evaluate(string_alg)
    # print("t_string_alg", t_string_alg)
    # interp_t_string_alg = t_string_alg.evaluate()
    # print("evaluate", interp_t_string_alg)
    # inner_term = interp_t_string_alg.inner_term
    # print("inner", inner_term)
    # string_output = inner_term.evaluate()
    # print("output", string_output)
    #
    # print("***************")
    # t_am_alg = t.evaluate(am_alg)
    # print("t_am_alg", t_am_alg)
    # # interp_t_am_alg = t_am_alg.evaluate()
    # # print("evaluate", interp_t_am_alg)
    # # inner_term = interp_t_am_alg.inner_term
    # # print("inner", inner_term)
    # # g_output = inner_term.evaluate()
    # # print("output", g_output, type(g_output))
    # print("******************")
    #
    # print("slept term", t2.evaluate(am_alg))
    # print("***************")

    # each algebra is mapped to (inner_op, prepare, reverse)
    inners_s = {
        string_alg: InnerAlgebraInstructions("concat_left", reverse=True),
        am_alg: InnerAlgebraInstructions("App_S"),
    }

    inners_o = {
        string_alg: InnerAlgebraInstructions("concat_right"),
        am_alg: InnerAlgebraInstructions("App_O"),
    }

    inners_op1 = {
        string_alg: InnerAlgebraInstructions("concat_left", reverse=True),
        am_alg: InnerAlgebraInstructions("App_OP1"),
    }

    inners_op2 = {
        string_alg: InnerAlgebraInstructions("concat_right"),
        am_alg: InnerAlgebraInstructions("App_OP2"),
    }

    inners_atb_op1 = {
        string_alg: InnerAlgebraInstructions("concat_left", "excorporation", reverse=True),
        am_alg: InnerAlgebraInstructions("App_OP1"),
    }

    inners_atb_op2 = {
        string_alg: InnerAlgebraInstructions("concat_right", "excorporation"),
        am_alg: InnerAlgebraInstructions("App_OP2"),
    }

    inners_r = {
        string_alg: InnerAlgebraInstructions("concat_right"),
    }

    inners_hm = {
        string_alg: InnerAlgebraInstructions("concat_right", "prefix"),
    }

    inners_atb = {
        string_alg: InnerAlgebraInstructions("concat_right", "hm_atb"),
        am_alg: InnerAlgebraInstructions("App_OP1")
    }

    # add triple algebra
    for ops in [inners_s, inners_r, inners_hm, inners_atb, inners_atb_op2, inners_atb_op1, inners_op1, inners_op2, inners_o]:
        ops[triple_alg] = copy(ops[string_alg])
        ops[triple_alg].reverse = False

    merge_s = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_s,
                                            name="merge_S")
    merge_o = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_o,
                                            name="merge_O")
    merge_r = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_r,
                                            name="merge_right")
    merge_op1 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                              inner_ops=inners_op1,
                                              name="merge_op1")
    merge_op2 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                              inner_ops=inners_op2,
                                              name="merge_op2")
    merge2 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge2,
                                           to_slot=Slot(A),
                                           name="merge_A")
    move1 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.move1,
                                          from_slot=A,
                                          inner_ops=inners_s,
                                          name="move_A")
    merge_hm = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1, inner_ops=inners_hm,
                                             name="merge_hm")

    merge_atb = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                              inner_ops=inners_atb,
                                              name="merge_atb")

    merge_atb_op1 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                                  inner_ops=inners_atb_op1,
                                                  name="merge_atb_op1")
    merge_atb_op2 = MinimalistFunctionSynchronous(minimalist_algebra=mg, minimalist_function=mg.merge1,
                                                  inner_ops=inners_atb_op2,
                                                  name="merge_atb_op2")

    # has ATB head movement
    tree_atb_hm = SynchronousTerm(merge_hm, [
        q,
        SynchronousTerm(merge_atb, [
            SynchronousTerm(merge_atb_op2, [
                and_term,
                SynchronousTerm(move1, [
                    SynchronousTerm(merge_r, [
                        did,
                        SynchronousTerm(merge2, [
                            sleep,
                            SynchronousTerm(merge_r,
                                            [
                                                the,
                                                cat
                                            ])
                        ])
                    ])
                ])
            ]),
            SynchronousTerm(move1, [
                SynchronousTerm(merge_r, [
                    did,
                    SynchronousTerm(merge2, [
                        dream,
                        SynchronousTerm(merge_r,
                                        [
                                            the,
                                            dog
                                        ])
                    ])
                ])
            ])
        ])
    ])

    # has ATB phrasal movement
    tree_atb = SynchronousTerm(move1,
                               [
                                   SynchronousTerm(merge_r,
                                                   [
                                                       past,
                                                       SynchronousTerm(merge_op1,
                                                                       [
                                                                           SynchronousTerm(merge_op2, [
                                                                               and_term,
                                                                               SynchronousTerm(merge2, [
                                                                                   slept,
                                                                                   SynchronousTerm(merge_r,
                                                                                                   [
                                                                                                       the,
                                                                                                       cat
                                                                                                   ])
                                                                               ])
                                                                           ]),
                                                                           SynchronousTerm(merge2, [
                                                                               dreamt,
                                                                               SynchronousTerm(merge_r,
                                                                                               [
                                                                                                   the,
                                                                                                   cat
                                                                                               ])
                                                                           ])
                                                                       ]

                                                                       )
                                                   ])
                               ])

    # term for "the cat tried to sleep"
    t3 = SynchronousTerm(move1,
                         [SynchronousTerm(
                             merge_o,
                             [tried,
                              SynchronousTerm(
                                  merge_r,
                                  [to,
                                   SynchronousTerm(
                                       merge2,
                                       [sleep,
                                        SynchronousTerm(
                                            merge_r,
                                            [the, cat]
                                        )
                                        ]
                                   )
                                   ]
                              )
                              ]
                         )
                         ]
                         )

    # See all three interpretations
    # print("SMG term ###################")
    # print(tree_atb_hm)

    # print("\n***** AM *********")
    # print(am_alg.ops)
    # e = t3.evaluate(am_alg)
    # print(e)
    # inner = e.inner_term
    # print(inner)
    # g = inner.evaluate()
    # print(g)
    # print(tree_atb_hm.spellout(am_alg))

    # print(t3.interp(am_alg).spellout())
    #
    # print("\n*********String********")
    # print(t3.interp(string_alg).spellout())
    #
    # print("\n*********Triples********")
    # print(t3.interp(triple_alg).spellout())
    #
    # print("\n********* Addresses ****************")
    # print(t3.interp(addresses_alg).spellout())
    #
    # print(tried.to_minimalist_term(addresses_alg).parent.function)

    print("\n********* Adding addresses as we go *************")
    # new copy of the addresses algebra
    # address_alg = HMTripleAlgebra("adding addresses algebra", component_type=list, addresses=True)
    address_prepare = HMAddressedTriplesPreparePackages()


    # print("starting ops of root:", t3.parent.inner_ops)

    def add_addresses(term: SynchronousTerm, address):
        """
        Add addresses triples interpretation
        # TODO currently has algebras hard-coded. Could be fine if only used in mgbank2algebra.
        """
        logger.debug("term: " + str(term))

        if term.is_leaf():
            # for these purposes, we don't want the silent heads to have addresses
            if len(term.spellout(triple_alg)) == 0:
                new_object = Triple(component_type=list)
            else:
                new_object = Triple([[address]])

            # If there are no specified inner ops for any existing interpretations
            # (because they'll just use the default), initialise an empty dict
            if term.parent.inner_ops is None:
                term.parent.inner_ops = {}

            # assert isinstance(term.parent.function, Expression), f"{term.parent.function} is not an Expression"
            if triple_alg in term.parent.inner_ops:
                new_name = term.parent.inner_ops[triple_alg].op_name
            else:
                new_name = term.parent.name
            if address_alg in term.parent.inner_ops:
                # we need to copy this leaf
                new_inner_ops = {}
                for a in term.parent.inner_ops:
                    if a != address_alg:
                        new_inner_ops[a] = deepcopy(term.parent.inner_ops[a])
                # algebra functions need unique names so we add the address to the copy
                new_name = term.parent.name  # + "_" + address
                new_function = MinimalistFunctionSynchronous(minimalist_algebra=term.parent.minimalist_algebra,
                                                             minimalist_function=term.parent.minimalist_function,
                                                             inner_ops=new_inner_ops,
                                                             conj=term.parent.conj,
                                                             name=new_name)
                new_function.inner_ops[address_alg] = InnerAlgebraInstructions(new_name, leaf_object=new_object)
                return SynchronousTerm(new_function), address

            term.parent.inner_ops[address_alg] = InnerAlgebraInstructions(new_name, leaf_object=new_object)
            return term, address
        else:
            if term.parent.inner_ops is not None:
                # logger.debug(f"\n############# Inner Ops {term.parent.inner_ops} #####################\n")
                assert triple_alg in term.parent.inner_ops
                # these are both HM algebras so their function names etc are the same
                term.parent.inner_ops[address_alg] = term.parent.inner_ops[triple_alg]
            new_pairs = [add_addresses(kid, address + str(i)) for (i, kid) in
                         enumerate(term.children)]
            # we just want the new_term, not the subterm of the other term
            term.children = [pair[0] for pair in new_pairs]
            # return the current updated subterm and other subterm
            return term, address


    t = t3
    print("triple:", t.interp(triple_alg))
    mg.inner_algebras[address_alg] = address_prepare
    add_addresses(t, "")
    #
    # print(address_alg.meta)
    #
    # print(tree_atb_hm)
    #
    # print()
    #
    address_output = t.spellout(address_alg)
    triple_output = t.spellout(triple_alg).split()
    print(len(triple_output) == len(address_output))
    print(triple_output)
    print(address_output)


    def find_index(address, sentence):
        """
        Look through a list of lists of addresses and fine the index in the main list that contains the given address.
        Used to find where in the sentence a given leaf in the MG term actually ends up.
        Copying from ATB movement yields some words with multiple sources,
         which is why the "sentence" is a list of lists of addresses instead of just a list of addresses.
        @param address: any, but used for str (tree address as string of integers).
        @param sentence: list of lists of addresses.
        @return: int, the index where the address was found.
        """
        for i, addresses in enumerate(sentence):
            if address in addresses:
                return i


    interval_algebra = HMIntervalPairsAlgebra()
    interval_prepare = IntervalPairPrepare()
    interval_algebra.add_constant_maker()
    mg.inner_algebras[interval_algebra] = interval_prepare


    def add_intervals(term: SynchronousTerm, sentence):
        """
        Given a SynchronousTerm that includes an address triple algebra,
        add an Interval algebra with the indices in the string.
        Called recursively.
        @param term: SynchronousTerm that includes an address triple algebra.
        @param sentence: spellout of the term using the address triple algebra, so list of lists of strings.
        @return: updated term, sentence
        """
        if term.is_leaf():
            address_list = term.parent.inner_ops[address_alg].leaf_object.head
            if len(address_list) > 0:
                # these are always unary list in the tree, because it's the address of the leaf itself.
                start = find_index(address_list[0][0], sentence)
                name = str(start)
            else:
                start = None
                name = term.parent.name
            term.parent.inner_ops[interval_algebra] = InnerAlgebraInstructions(name, leaf_object=PairItem(start))
            return term, sentence
        else:
            if term.parent.inner_ops is not None and triple_alg in term.parent.inner_ops:
                # these are both HM algebras, so their ops and prepares have the same names
                term.parent.inner_ops[interval_algebra] = term.parent.inner_ops[triple_alg]
            new_pairs = [add_intervals(kid, sentence) for kid in term.children]
            # we just want the new_term, not the subterm of the other term
            term.children = [pair[0] for pair in new_pairs]
            # return the current updated subterm and other subterm
            return term, sentence



    add_intervals(t, address_output)
    bare_tree = t.interp(string_alg).inner_term
    # bare_tree.to_nltk_tree().draw()
    sentence = t.spellout(string_alg)
    print(sentence)

    print(t.interp(am_alg).inner_term.evaluate().to_graphviz())

    print(t.spellout(interval_algebra))


    def mg_op2mg_op(op: MinimalistFunction, out_alg: HMAlgebra,
                    out_prepare: PreparePackagesHM):
        """
        Turns MinimalistFunction over one algebra into the same one over another algebra, if the inner_op and prepare
         functions of the original function are also in out_alg.
        @param out_prepare: the prepare packages for the output algebra
        @param op: operation to transform.
        @param out_alg: the inner algebra of the new MinimalistFunction.
        @return: MinimalistFunction with inner algebra out_alg, otherwise just like op.
        """

        return MinimalistFunction(op.mg_function_type,
                                     name=op.name,
                                     inner_op=getattr(out_alg, op.inner_op.__name__) if op.inner_op else None,
                                     from_slot=op.from_slot,
                                     to_slot=op.to_slot,
                                     prepare=getattr(out_prepare, op.prepare.__name__) if op.prepare else None,
                                     adjoin=op.adjoin,

                                     )


    def partial_result(term: SynchronousTerm, inner_algebra: Algebra, mover_type=DSMCMovers):
        expr = term.interp(inner_algebra)
        inner_result = expr.inner_term.evaluate()
        if term.is_leaf():
            return Expression(inner_result, mover_type(), expr.mg_type)
        else:
            for slot_name in expr.movers.mover_dict:
                slot = expr.movers.mover_dict[slot_name]
                slot.contents = slot.contents.evaluate()
            return Expression(inner_result, expr.movers, mg_type=expr.mg_type)

    def get_ops_and_parse_items(term: SynchronousTerm, op_state_copy_list):
        """
        Given a synchronous term with an interval pair algebra interpretation,
        get a list of triple: (minimalist op, children as parse items, list of how many copies we need of each child)
        @param term: SynchronousTerm.
        @param op_state_copy_list: list being updated.
        @param address_sequence: output of the address algebra for the whole sentence.
        @return: list of triples of (SynchronousMinimalistFunction,
                                        list of children evaluated into Expressions over PairItems,
                                        list of how many times each child should be copied for ATB reasons)
        """
        if not term.is_leaf():
            op = term.parent
            # results = [partial_result(child, interval_algebra) for child in term.children]

            # use the output of the addresses algebra to see if we need to copy any leaves
            # copy_children = []
            # for child in term.children:  # results:
            #     if child.is_leaf():
            #         # this is the index of the word in the sentence
            #
            #         head = interval_prepare.get_head(child.interp(interval_algebra).inner_term)
            #         print("head:", head)
            #         try:
            #             start = int(head.parent.name)
            #             # if this word has more than one address, copy it that many times
            #             copies = len(address_sequence[start]) - 1
            #         except ValueError:
            #             copies = 0  # silent
            #     else:
            #         copies = 0  # we only copy leaves, so the intermediate results can be built again
            #     copy_children.append(copies)

            op_state_copy_list.append((op, term.children))
            for child in term.children:
                get_ops_and_parse_items(child, op_state_copy_list)
        return term, op_state_copy_list


    def find_child_in_stack(child, stack):
        if child.is_lexical() and len(child.inner_term.parent.function) == 0:
            return child.inner_term.parent.name
        try:
            return stack.index(child)
        except ValueError:
            found = [i for i, stack_item in enumerate(stack) if child.equal_modulo_conj(stack_item)]
            if len(found) > 0:
                return found[0]
            else:
                print(f"didn't find {child} in {stack}")


    _, acts = get_ops_and_parse_items(t, [])
    acts.reverse()
    def mark_duplicates(l, new_list):
        if len(l) == 1:
            new_list.append((l[0], 0))
            return new_list
        else:
            item = l[0]
            c = l[1:].count(item)
            print(f"found {c} copies of {item}")
            new_list.append((item, c))
            return mark_duplicates(l[1:], new_list)

    x = mark_duplicates(acts, [])
    print(x)

    # initialise stack
    stack = [Expression(interval_algebra.make_leaf(i), DSMCMovers()) for i in range(len(address_output))]
    print(stack)
    actions = []
    for (op, children), copies in x:
        print("handling", op, children, copies)
        print("Stack:", stack)
        interpreted_children = []
        for child in children:
            kid = child.interp(interval_algebra)
            if kid.inner_term.parent.name == "None":
                kid.inner_term.parent.name = child.interp(triple_alg).inner_term.parent.name
            interpreted_children.append(kid)

        inner_terms = [child.inner_term for child in interpreted_children]
        # for term in inner_terms:
        #     for item in stack:
        #         # print(item.inner_term.parent, term.parent)
                # print(item.inner_term.parent.name == term.parent.name)
                # print(item.inner_term.parent.function, term.parent.function)
                # print(item.inner_term.parent.function == term.parent.function)
        child_stack_indices = [find_child_in_stack(child, stack) for child in interpreted_children]

        print("found", child_stack_indices)
        actions.append((op, child_stack_indices))

        result = op.single_function(interval_algebra).function([child.interp(interval_algebra) for child in children])
        print(result)
        if copies > 0:
            print("might copy")
            for i, child in enumerate(children):
                if child.is_leaf():
                    print("copy", child)
                    stack += [stack[child_stack_indices[i]]] * copies
        if not isinstance(child_stack_indices[0], int):
            ind = child_stack_indices[1]
        else:
            ind = child_stack_indices[0]
            if len(children) > 1:
                stack[child_stack_indices[1]] = None
        stack[ind] = result


    print(stack)
    print(actions)

    def write_actions_to_file(sentence, action_list, path_to_file):
        # print(os.getcwd())
        with open(path_to_file, "a") as f:
            s = f"{sentence}\t"
            for op, args in action_list:
                s += f"{op.name} "
                for arg in args:
                    s += f"{arg} "
            s += "\n"
            f.write(s)

    toy_data_path = "data/processed/shift_reduce/toy/data.txt"
    write_actions_to_file(sentence, actions, toy_data_path)




