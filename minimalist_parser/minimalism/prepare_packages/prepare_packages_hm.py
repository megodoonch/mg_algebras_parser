from abc import ABC

from ...algebras.algebra import AlgebraTerm, AlgebraOp
from ...algebras.hm_algebra import HMAlgebra, HMError
from .prepare_packages import PreparePackages

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class PreparePackagesHM(PreparePackages, ABC):
    """
    Abstract class for prepare functions used for head movement.
    suffix, prefix, hm_atb (across-the-board head movement), excorporation.
    """
    def __init__(self, name, inner_algebra: HMAlgebra):
        super().__init__(name, inner_algebra)
        self.packages = {self.suffix, self.prefix, self.hm_atb, self.excorporation}

    @staticmethod
    def simple(term_1, term_2):
        return term_1, term_2

    def suffix(self, functor: AlgebraTerm, argument: AlgebraTerm):
        """
        Prepare Package
        Applies HM to prefix to the heads, but keeps rest separate
        @param functor: TermItem
        @param argument: TermItem
        @return: TermItem pair
        """

        updated_argument, argument_head = self.extract_head(argument)
        op = self.inner_algebra.get_op_by_name("suffix")
        new_functor = self.add_affix(functor, argument_head, op)

        return new_functor, updated_argument

    def prefix(self, functor: AlgebraTerm, argument: AlgebraTerm):
        """
        Prepare Package
        Applies HM to suffix to the heads, but keeps rest separate
        @param functor: AlgebraTerm
        @param argument: AlgebraTerm
        @return: AlgebraTerm pair
        """
        updated_argument, argument_head = self.extract_head(argument)
        op = self.inner_algebra.get_op_by_name("prefix")
        new_functor = self.add_affix(functor, argument_head, op)

        return new_functor, updated_argument

    # def hm_atb(self, term_1, term_2):
    #     raise NotImplementedError

    def __repr__(self):
        return self.name

    # moved from hm_algebra
    # utility functions for messing with the terms
    def is_head_subterm(self, term):
        """
        A method for identifying heads
        Searching downward and to the left, we'll find the root of the head subterm when we first find a
            node not labelled with one of the non-head operations or we hit a leaf
        @param term: AlgebraTerm
        @return: bool (true if this is the head's subterm)
        """
        return term.is_leaf() or term.parent.name not in self.inner_algebra.syntax_op_names

    def add_affix(self, term: AlgebraTerm, affix: AlgebraTerm, affixation_function):
        """
        Tree homomorphism that adds affix to right edge of the head's subterm using the given affixation_function
        @param affixation_function:
        @param affix: AlgebraTerm: the affix to add
        @param term: the AlgebraTerm to update
        @return: updated AlgebraTerm
        """
        # heads are leftward leaves or leftward subterms not rooted by a concat operation (e.g. another suffix)
        assert isinstance(term, AlgebraTerm), f"{term} should be an AlgebraTerm"
        if self.is_head_subterm(term):
            logger.debug(f"Head subterm {term}")
            return AlgebraTerm(affixation_function, children=[term, affix])
        else:
            if len(term.children) > 1:
                remaining_children = term.children[1:]
            else:
                remaining_children = []
            new_left_daughter = self.add_affix(term.children[0], affix, affixation_function)
            return AlgebraTerm(term.parent, [new_left_daughter] + remaining_children)

    def remove_head(self, term: AlgebraTerm):
        """
        Finds head by looking down the left branches until we hit a leaf or a non-concat function
        Returns the term an empty item in place of the head subterm
        @param term: the AlgebraTerm to update
        @return: the updated AlgebraTerm paired with the extracted head
        """
        if self.is_head_subterm(term):
            # logging.debug(f"Removing head {term} in algebra {self.name}, replacing with {self.empty_object}")
            # logging.debug(f"as empty leaf: {self.empty_leaf.parent.function}")
            ret = self.inner_algebra.empty_leaf
            assert ret is not None, f"{self.inner_algebra} has no empty leaf"
            return ret
        else:
            if len(term.children) > 1:
                remaining_children = term.children[1:]
            else:
                remaining_children = []
            new_left_daughter = self.remove_head(term.children[0])

            return AlgebraTerm(term.parent, [new_left_daughter] + remaining_children)

    def get_head(self, term: AlgebraTerm):
        """
        Finds head by looking down the left branches until we hit a leaf or a non-concat function
        @param term: the AlgebraTerm to update
        @return: the updated AlgebraTerm
        """
        if self.is_head_subterm(term):
            return term
        else:
            return self.get_head(term.children[0])

    def extract_head(self, term: AlgebraTerm):
        """
        Returns a pair of the remainder and the head once the head is removed.
        @param term: input term.
        @return: pair of algebra terms
        """
        return self.remove_head(term), self.get_head(term)

    def excorporation(self, functor: AlgebraTerm, other: AlgebraTerm):
        """
        Prepare Package
        New head is old head, and shove functor's head over to the right.
        @param functor AlgebraTerm
        @param other AlgebraTerm
        @return: updated AlgebraTerm that will evaluate to have new functor with the other.head,
                    and new argument with functor.head + other.left + other.right
        """
        updated_other, other_head = self.extract_head(other)
        updated_functor, functor_head = self.extract_head(functor)
        new_other = AlgebraTerm(self.inner_algebra.ops["concat_right"], [functor_head, updated_other])

        return other_head, new_other

    def hm_atb(self, functor: AlgebraTerm, other: AlgebraTerm):
        """
        Prepare Package
        Implements across-the-board head movement
        Merge to the left. Heads must be identical, and head type must be conj.
        If we have addresses, combines addresses on identical words
        @param other: Item: the selectee
        @param functor: Item: the selector
        @return: pair of Items: the head, other_rest + functor_rest
                                i.e. (_, h, _) and
                                (other.left + other.right + functor.left
                                 + functor.right)
        """
        updated_other, other_head = self.extract_head(other)
        updated_functor, functor_head = self.extract_head(functor)

        # we need to do string checking, so we evaluate the heads
        functor_head_string = functor_head.evaluate().collapse()
        other_head_string = other_head.evaluate().collapse()

        if not functor_head_string == other_head_string:
            raise ATBError(f"Can only do ATB HM if heads are identical"
                           f" ({functor_head_string} vs {other_head_string})")
        # else:
        #     new_head_string = functor_head_string

        rest = AlgebraTerm(self.inner_algebra.ops["concat_right"], [updated_other, updated_functor])

        return functor_head, rest
        #self.inner_algebra.make_leaf(head=, function=new_head_string), rest


class ATBError(HMError):
    """
    Exception raised when trying to do ATB movement to non-identical strings
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = "ATBError: " + message
