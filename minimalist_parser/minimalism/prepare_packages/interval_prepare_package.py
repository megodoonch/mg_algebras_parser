from ...algebras.algebra_objects.intervals import Pair, Interval, PairItem
from .prepare_packages_hm import ATBError
from ...minimalism.prepare_packages.prepare_packages_hm import PreparePackagesHM
from ...algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra


class IntervalPairPrepare(PreparePackagesHM):
    def __init__(self):
        super().__init__("HM Interval Pairs Prepare Packages", HMIntervalPairsAlgebra())
    # def prefix(self, functor: PairItem, argument: PairItem):
    #     """
    #     Applies HM to prefix to the heads, but keeps rest separate
    #     @param functor: PairItem
    #     @param argument: PairItem
    #     @return: PairItem pair
    #     """
    #     new_functor = argument.head + functor.head
    #     new_functor - argument.rest  # raises IntervalError if items overlap
    #
    #     # new functor may itself still need to HM, but argument already did
    #     return PairItem(new_functor, lexical=True, must_hm=functor.must_hm),\
    #         PairItem(argument.rest, lexical=False, must_hm=False)
    #
    # def suffix(self, functor, argument):
    #     """
    #      Applies HM to suffix to the heads, but keeps rest separate
    #      @param functor: Item
    #      @param argument: Item
    #      @return: PairItem pair
    #      """
    #
    #     updated_argument, argument_head = self.inner_algebra.extract_head(argument)
    #
    #
    #     new_functor = functor.head + argument.head
    #     new_functor - argument.rest  # raises IntervalError if items overlap
    #     # new functor may itself still need to HM, but argument already did
    #     return PairItem(new_functor, lexical=True, must_hm=functor.must_hm),\
    #         PairItem(argument.rest, lexical=False, must_hm=False)

    # def hm_atb(self, functor, other):
    #     """
    #     Implements across-the-board head movement
    #     Heads must be identical, and head type must be conj.
    #     @param other: PairItem
    #     @param functor: PairItem
    #     @return: pair of Items: the head, other_rest + functor_rest
    #                         i.e. (h, _) and
    #                         (other.left + other.right + functor.left
    #                          + functor.right)
    #     """
    #     if not other.can_hm and functor.can_hm:
    #         raise ATBError("Can only do ATB HM when both can HM")
    #
    #     if not functor.head == other.head:
    #         raise ATBError(
    #             f"Can only do ATB HM if heads are identical"
    #             f" ({functor.head} vs {other.head})")
    #
    #     rest = other.rest + functor.rest
    #     return PairItem(Pair(functor.head, Interval.empty_interval()), must_hm=True, lexical=True), \
    #         PairItem(rest, lexical=False, must_hm=False)
