"""
HM algebra for MGBank

algebra objects are Intervals and pairs of Intervals

Not using AL since this is for MGBank

"""
from .hm_algebra import *

from .algebra_objects.intervals import Interval, Pair, PairItem

# logging: change VERBOSE to False to stop printing everything
# VERBOSE = False
VERBOSE = True

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug


class HMIntervalPairsAlgebra(HMAlgebra, ABC):
    """
    Algebra over PairItems, which are Intervals and pairs thereof
    e.g. (2,4) or ((2,4),(5,6))
    """

    def __init__(self, name=None):
        name = name or "Interval Pair Algebra"
        super().__init__(name, PairItem)

    def concat_left(self, args: list[PairItem]):
        """
        Concatenate args[1] to the left of args[0]
        If functor is lexical, we can leave a gap, since HM can apply later
        Otherwise, we concatenate to the rest, if possible, or concat all 3
        Raises errors when impossible
        @param args: list of two Items
        @return: Item
        """
        functor, other = args[0], args[1]
        log("concatenate {} to the left of {}".format(other, functor))

        # other needs to be a single span; otherwise operation fails
        other_int = other.collapse()

        if functor.lexical:
            # must hm iff other isn't immediately to the left of head
            hm = not other_int.is_immediately_left_of(functor.head)
            new_item = Pair(functor.head, other_int)
            logger.debug(f"Not tupled. Created {new_item}.")
        else:
            if functor.tupled:
                # now that we have a pair, we have to keep
                # left-concatenating leftward
                if other_int < functor.rest:
                    # cases where we concat with rest: head must move
                    # or rest is in fact already leftward material
                    if functor.must_hm or functor.rest < functor.head:
                        hm = functor.must_hm  # keep HM type
                        new_item = Pair(functor.head,
                                        other_int + functor.rest)
                    else:
                        # otherwise, we already have rightward material, and head is
                        # already adjacent to it, so concat everything
                        hm = False
                        new_item = other_int + functor.head + functor.rest
                else:
                    raise HMError(
                        "Error in concat_left: can't left-concat"
                        " to the right of a derived item")
            else:
                # derived Interval can't be messed with;
                # just concat left if possible
                hm = False
                new_item = other_int + functor.span
        ret = PairItem(new_item, lexical=False, must_hm=hm)
        logger.debug(f"created PairItem: {ret}")
        return ret

    def concat_right(self, args: list[PairItem]):
        """
        Concatenate args[1] to the right of args[0]
        If functor is lexical, we can leave a gap, since HM can apply later
        Otherwise, we concatenate to the rest, if possible, or concat all 3
        Raises errors when impossible
        @param args: list of two Items
        @return: Item
        """
        functor, other = args[0], args[1]
        log("concatenate {} to the right of {}".format(other, functor))

        # other needs to be a single span; otherwise operation fails
        other_int = other.collapse()

        if functor.lexical:
            # must hm iff other isn't immediately to the right of head
            hm = not functor.span.is_immediately_left_of(other_int)
            new_item = Pair(functor.span, other_int)
        else:
            if functor.tupled:
                # now that we have a pair, we have to keep right-concating rightward
                if functor.rest < other_int:
                    # cases where we concat with rest: head must move
                    # or rest is in fact already leftward material
                    if functor.must_hm or functor.head < functor.rest:
                        hm = functor.must_hm  # keep HM type
                        new_item = Pair(functor.head, functor.rest + other_int)

                    else:
                        # otherwise, we already have leftward material, and head
                        # is already adjacent to it, so concat everything
                        hm = False
                        new_item = functor.rest + functor.head + other_int
                else:
                    raise HMError(
                        f"Error in concat_right: can't right-concat"
                        f" to the left of a derived item ({functor}"
                        f" selecting {other_int})")
            else:
                # derived Interval can't be messed with;
                # just concat right if poss.
                hm = False
                new_item = functor.span + other_int
        ret = PairItem(new_item, lexical=False, must_hm=hm)
        logger.debug(f"created PairItem: {ret}")
        return ret

    def suffix(self, args: list[PairItem]):
        logger.debug(f"suffix: combining {args[0]} + {args[1]}")
        assert len(args) == 2, "suffix requires exactly 2 arguments"
        assert type(args[0]) == PairItem and type(args[1]) == PairItem, f"suffix requires PairItems," \
                                                                        f" but got {type(args[0])} and {type(args[1])}"
        assert not args[0].tupled, "first argument of suffix must be lexical: here, an Interval"
        assert args[1].can_hm, f"second argument to suffix can't undergo head movement: {args[1]}"
        return args[0].concat_suffix(args[1].head)

    def prefix(self, args: list[PairItem]):
        logger.debug(f"prefix: combining {args[1]} + {args[0]}")
        assert len(args) == 2, "prefix requires exactly 2 arguments"
        assert type(args[0]) == PairItem and type(args[1]) == PairItem, f"prefix requires PairItems," \
                                                                        f" but got {type(args[0])} and {type(args[1])}"
        assert args[1].can_hm, f"second argument to prefix can't undergo head movement: {args[1]}"

        # if this came from l_merge_lex, then this is not lexical, but prefix is still allowed.
        # in this case, it'll be of the form Pair(empty),(head), even though it really should be Interval(head)
        # Treat the 'rest' as if it's the head.
        if not args[0].lexical:
            logger.info(f"using prefix with non-lexical head: {args[0]}")
            # assert not args[0].tupled, "first argument of prefix must be lexical: here, an Interval"
            ret = args[0].concat_prefix(args[1].head)
        elif len(args[0].head) == 0:
            # use concat_suffix because then we can explicitly apply it to .rest.
            ret = args[1].concat_suffix(args[0].rest)
        else:
            raise AlgebraError(f"first argument of prefix must be lexical: here, an Interval,"
                               f" or, in the weird l_merge_lex case, have an empty head.")
        logger.debug(f"new head: {ret}")
        return ret

    def default_constant_maker(self, word, label=None):
        return AlgebraOp(str(word), PairItem(word))


if __name__ == "__main__":
    a = HMIntervalPairsAlgebra()
    print(a)

    print(a.empty_leaf)

    the = PairItem(Interval(0))
    print(the.can_hm)

    cat = PairItem(Interval(1))
    print(cat)

    slept = PairItem(Interval(2))
    print(slept)

    conjunct = PairItem(Interval(2))
    dog = PairItem(Interval(3))

    the_cat = a.concat_right([the, cat])
    print(the_cat)

    the_cat_slept = a.concat_left([slept, the_cat])

    print(the_cat_slept)

    the_slept = a.concat_right([the, slept])

    print(the_slept)
    print(the_slept.can_hm)

    the_cat_slept = a.concat_left([the_slept, cat])

    print(the_cat_slept)
