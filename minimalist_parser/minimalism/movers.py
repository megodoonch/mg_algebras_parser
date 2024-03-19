"""
For mover storage in a minimalist algebra based on MGBank

class Slot: a to_slot for storing a mover
class Movers: contains a dict for storing movers, and controls SMC
subclasses of Movers:
    DSMCMovers: implements the complicated DSMC stuff with A movers
        blocking "EPP" movers, which otherwise may co-occur with up to 2
    DSMCAddressedMovers: like DSMCMovers but words have the form word.address,
        where address is a sequence of 0's and 1's. For ATB purposes,
        word1.add1 == word2.add2 iff word1 == word2 and the new word
        is word1.add1.add2
    ListMovers: WARNING: this makes the MG type-0 because it includes Slots with
        the property multiple, where a list of movers is allowed.
        Currently does not implement the DSMC A/EPP stuff but it could
        In a multiple slot, movers are added to the end, but can be
        removed by index


function combine_movers combines 2 Movers, respecting the SMC
        and allowing ATB movement
function combine_movers_addresses is for the string triple algebra when it has
    addresses on the node labels. We want to treat the movers the as the same if
    they're the esame except for their addresses ,and we want to add both the
    addresses to the words in the tuples. Called by combine_movers.
function atb_combine_slot combines 2 words if they are strings ending in a dot
    followed by a sequence of 0's and 1's. If the words are the same except for the
    addresses, it adds both addresses to the string
"""
from minimalist_parser.algebras.hm_triple_algebra import combine_leaf_addresses
from ..algebras.algebra_objects.triples import Triple
from .mg_errors import SMCViolation, ExistenceError, MGError
from copy import deepcopy, copy
import re


import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug


address_sep = "ADDR_SEP"


class Slot:
    """
    A mover to_slot
    properties a and multiple control its behaviour regarding the DSMC: A and EPP
     movers can't co-occur, but up to 2 EPP features can co-occur
    Attributes:
        name: to_slot name, such as A
        a: is an A-to_slot
        multiple: is an EPP-to_slot
        contents: mover(s)
    """

    def __init__(self, name: str, contents=None, a=False, epp=False,
                 multiple=False):
        self.name = name
        self.a = a
        self.epp = epp
        self.multiple = multiple
        if self.multiple and contents is None:
            self.contents = []
        else:
            self.contents = contents

    def __copy__(self):
        return Slot(self.name, contents=copy(self.contents),
                    a=self.a, epp=self.epp, multiple=self.multiple)

    def __repr__(self):
        ret = self.name
        if self.a:
            ret += "_A"
        if self.epp:
            ret += "_EPP"
        if self.contents:
            ret += f":{self.contents}"
        return ret

    def __eq__(self, o) -> bool:
        logger.debug(f"{self.name} vs {o.name} ({self.name == o.name})\n{self.multiple} vs {o.multiple} ({self.multiple == o.multiple})\n{self.epp} vs {o.epp} ({self.epp == o.epp})\n{self.a} vs {o.a} ({self.a == o.a})\n ({self.contents == o.contents}) {self.contents} vs {o.contents}")
        return type(o) == Slot and self.contents == o.contents and \
               self.name == o.name and self.multiple == o.multiple and \
               self.a == o.a and self.epp == o.epp

    def combine_multiple_movers(self, epp_movers):
        """
        If multiple, concat new movers
        @param epp_movers:
        @return:
        """
        if self.multiple:
            self.contents += epp_movers

        else:
            raise MGError(f"Can't add mover to filled {self.name} to_slot")

    def add_contents(self, contents):
        if self.multiple:
            self.contents.append(contents)

        else:
            self.contents = contents


class Movers:
    """
    Implements a basic minimalist mover store
    SMC implemented by not allowing more than one mover per to_slot
    No SpIC
    """

    def __init__(self, movers: dict = None):
        """
        Makes a new Movers object. If a mover_dict is supplied, checks for A and
         EPP movers and notes them in a_filled and epps_filled
        @param movers: dict from str to Slots containing objects of the
                        inner algebra (Items)
        """
        self.mover_dict = dict() if movers is None else movers

    def __repr__(self):
        movers = []
        for slot in self.mover_dict:
            movers.append(f"{self.mover_dict[slot]}")
        return f"{{{', '.join(movers)}}}"

    def __eq__(self, o) -> bool:
        return isinstance(o, type(self)) and self.mover_dict == o.mover_dict

    def add_mover(self, filled_slot: Slot):
        """
        Add a new mover
        @param filled_slot: a Slot
        @return: new Movers
        """
        if self.mover_dict.get(filled_slot.name, None) is None:
            # add the mover if there's room
            new_dict = deepcopy(self.mover_dict)  # copy the movers
            new_dict[filled_slot.name] = filled_slot
            return type(self)(movers=new_dict)

        else:
            raise SMCViolation(f"{filled_slot.name}"
                               f" already filled with {self.mover_dict[filled_slot.name]}; can't add {filled_slot}")

    def remove_mover(self, slot_name):
        """
        If there is a mover in the given to_slot, returns its contents,
         paired with a new Movers without that mover
        @param slot_name: str
        @return: Expression, Movers pair
        """
        if self.mover_dict.get(slot_name, None) is None:
            raise ExistenceError(f"No mover in {slot_name}")
        else:
            new_movers = deepcopy(self.mover_dict)
            mover = new_movers.pop(slot_name)  # remove the mover
            return mover, type(self)(new_movers)

    def atb_combine_movers(self, movers2, must_be_equal=False):
        """
        Across-the-board movement: returns movers2 if self <= movers2
        If must_be_equal, only if they're equal
        @param movers2: other Movers
        @param must_be_equal: bool
        @return: Movers
        """
        if must_be_equal and not self.mover_dict.keys() == movers2.mover_dict.keys():
            raise SMCViolation(f"ATB with merge: {self} != {movers2}")

        # if must be equal, this also works
        if self.mover_dict.items() <= movers2.mover_dict.items():
            return movers2
        else:
            raise SMCViolation(f"ATB: {self} not a subset of {movers2}")

    def combine_movers(self, movers2, adjoin=False):
        """
        Combines movers2 with self
        @param movers2: Movers
        @param adjoin: bool: if True, we'll allow combining movers if the adjunct's
                                movers is a subset of the head's
        @return: new Movers with combined mover dicts
        """
        if len(self.mover_dict) == 0:
            return movers2
        elif len(movers2.mover_dict) == 0:
            return self
        # adjoin: e2 is adjunct. Check for movers2 subset of self
        if adjoin:
            must_be_equal = False
        else:
            must_be_equal = True
        try:
            return self.atb_combine_movers(movers2, must_be_equal=must_be_equal)
        except SMCViolation:
            log(f"trying to combine {self} and {movers2}")
            new_movers = type(self)(deepcopy(self.mover_dict))
            for slot in movers2.mover_dict:
                log(f"adding {slot} to {new_movers}")
                new_movers = new_movers.add_mover(
                    copy(movers2.mover_dict[slot]))
            log(f"built {new_movers}")
            return new_movers


class DSMCMovers(Movers):
    """
    Implements a minimalist mover store
    For DSMC purposes, tracks whether the principal A to_slot is filled
     (e.g. with -case) -- because then we can't add any EPP movers --
     and how many EPP-slots are filled -- because if any are, no A-movers are
     allowed, and there is a maximum of 2 EPP-movers allowed
    Slots in the dict contain objects of the inner algebra (Items)
    Attributes:
        mover_dict: the movers in a dict from str to Slot
        a_filled: bool, whether the canonical A-to_slot is filled
        epps_filled: int, how many loc, num, pers, and epp movers we have
    """

    def __init__(self, movers=None, meta=None):
        """
        Makes a new Movers object. If a mover_dict is supplied, checks for A and
         EPP movers and notes them in a_filled and epps_filled
        @param meta: optional dict of extra information
        @param movers: dict from str to Slots containing objects of the
                        inner algebra (Items)
        """
        super().__init__(movers=movers)  # make self.mover_dict
        self.a_filled = False
        self.epps_filled = 0
        self.meta = meta if meta is not None else {}

        # extract info about movers
        for slot in self.mover_dict:
            mover = self.mover_dict[slot]
            if mover.a:
                self.a_filled = True
            elif mover.epp:
                self.epps_filled += 1

    def add_mover(self, filled_slot: Slot):
        """
        Add a new mover
        @param filled_slot: a Slot
        @return: new Movers
        """
        if (filled_slot.a or filled_slot.epp) and self.a_filled:
            raise SMCViolation(f"A already filled with {self.mover_dict[filled_slot.name]}; can't add {filled_slot}")
        if filled_slot.a and self.epps_filled > 0:
            raise SMCViolation(f"Can't add {filled_slot} when there"
                               f" are already {self.epps_filled} EPP-movers")
        if filled_slot.epp and self.epps_filled > 1:
            raise SMCViolation(f"Can't add {filled_slot} when there"
                               f" are already {self.epps_filled} EPP-movers")
        if self.mover_dict.get(filled_slot.name, None) is None:
            # add the mover if there's room
            new_dict = deepcopy(self.mover_dict)  # copy the movers
            new_dict[filled_slot.name] = filled_slot
            return type(self)(movers=new_dict)
        else:
            raise SMCViolation(f"{filled_slot.name}"
                               f" already filled; can't add {filled_slot}")

    def atb_combine_movers(self, movers2: Movers, must_be_equal=False):
        """
        Across-the-board movement: returns movers2 if self <= movers2
        If must_be_equal, only if they're equal
        @param movers2: other Movers
        @param must_be_equal: bool
        @return: Movers
        """
        if must_be_equal and not self.mover_dict.keys() == movers2.mover_dict.keys():
            raise SMCViolation(f"ATB with merge: {self} != {movers2}")

        # if must be equal, this also works
        if self.mover_dict.items() <= movers2.mover_dict.items():
            return movers2
        else:
            raise SMCViolation(f"ATB: {self} not a subset of {movers2}")


class DSMCAddressedMovers(DSMCMovers):
    """
    Implements a minimalist mover store
    movers have addresses, otherwise just like DSMCMovers
    For DSMC purposes, tracks whether the principal A to_slot is filled
     (e.g. with -case) -- because then we can't add any EPP movers --
     and how many EPP-slots are filled -- because if any are, no A-movers are
     allowed, and there is a maximum of 2 EPP-movers allowed
    Slots in the dict contain objects of the inner algebra (Items)
    Attributes:
        mover_dict: the movers in a dict from str to Slot
        a_filled: bool, whether the canonical A-to_slot is filled
        epps_filled: int, how many loc, num, pers, and epp movers we have
    """

    def __init__(self, movers=None):
        """
        Makes a new Movers object. If a mover_dict is supplied, checks for A and
         EPP movers and notes them in a_filled and epps_filled
        @param movers: dict from str to Slots containing objects of the
                        inner algebra (Items)
        """
        super().__init__(movers=movers)

    def atb_combine_movers(self, movers2, must_be_equal=False):
        """
        returns new Movers with combined dict where any overlapping movers can only
         vary by address. Addresses are combined in new mover objects.
        Called by combine_movers
        @param must_be_equal: bool; if true we require equality, modulo addresses,
         otherwise we require self to be a subset of movers2
        @param self: Movers
        @param movers2: Movers
        @return: new_movers Movers with combined addresses
        """
        if must_be_equal and \
                not self.mover_dict.keys() == movers2.mover_dict.keys():
            raise SMCViolation(f"ATB with merge: {self} != {movers2}")

        # if must be equal, this also works
        log(f"Checking if {self} is a subset of {movers2}, modulo addresses")
        if not self.mover_dict.keys() <= movers2.mover_dict.keys():
            raise SMCViolation(f"ATB with adjoin:"
                               f" {self} not a subset of {movers2}")
        new_movers = copy(movers2.mover_dict)
        for m in movers2.mover_dict:
            if m in self.mover_dict:
                # these are terms
                mover_2 = movers2.mover_dict[m].contents
                mover_1 = self.mover_dict[m].contents
                new_mover, _ = combine_leaf_addresses(mover_2, mover_1)
                logger.debug(f"new mover: {new_mover}")
                new_movers[m] = new_mover
        return movers2


class ListMovers(Movers):
    """
    Implements a minimalist mover store where some slots allow lists of movers

    Slots in the dict contain objects of the inner algebra (Items)
    Attributes:
        mover_dict: the movers in a dict from str to Slot

    """

    def __init__(self, movers=None):
        """
        Makes a new Movers object. If a mover_dict is supplied, checks for A and
         EPP movers and notes them in a_filled and epps_filled
        @param movers: dict from str to Slots containing objects of the
                        inner algebra (Items)
        """
        super().__init__(movers=movers)

    def add_mover(self, filled_slot: Slot):
        """
        Add a new mover
        @param filled_slot: a Slot
        @return: new Movers
        """
        if self.mover_dict.get(filled_slot.name, None) is None:
            # no to_slot of this name in mover_dict, so add it
            new_dict = deepcopy(self.mover_dict)  # copy the movers
            new_dict[filled_slot.name] = filled_slot
        elif filled_slot.multiple:
            # if multiples are allowed, we can add the mover lists together
            new_dict = deepcopy(self.mover_dict)  # copy the movers
            new_dict[filled_slot.name].combine_multiple_movers(filled_slot.contents)
        else:
            raise SMCViolation(f"{filled_slot.name}"
                               f" already filled; can't add {filled_slot}")
        return type(self)(movers=new_dict)

    def remove_mover(self, slot_name, index=0):
        """
        If there is a mover in the given to_slot, returns its contents,
         paired with a new Movers without that mover
        @param index:
        @param slot_name: str
        @return: Slot, Movers pair
        """
        mover = self.mover_dict.get(slot_name, None)
        if mover is None:
            raise ExistenceError(f"No mover in {slot_name}")
        elif mover.multiple:
            new_movers = deepcopy(self.mover_dict)
            mover = new_movers[slot_name].contents.pop(index)  # remove the mover
            if len(new_movers[slot_name].contents) == 0:  # delete slot if empty
                new_movers.pop(slot_name)
        else:
            new_movers = deepcopy(self.mover_dict)
            mover = new_movers.pop(slot_name)  # remove the mover
            mover = mover.contents
        return mover, type(self)(new_movers)

    def atb_combine_movers(self, movers2, must_be_equal: bool = False):
        """
        returns new Movers with combined dict where any overlapping movers can only
         vary by address. Addresses are combined in new mover objects.
        Called by combine_movers
        @param must_be_equal: bool; if true we require equality, modulo addresses,
         otherwise we require self to be a subset of movers2
        @param self: Movers
        @param movers2: Movers
        @return: new_movers Movers with combined addresses
        """
        if must_be_equal and \
                not self.mover_dict.keys() == movers2.mover_dict.keys():
            raise SMCViolation(f"ATB with merge: {self} != {movers2}")

        log(f"Checking if {self} is a subset of {movers2}")
        if not self.mover_dict.keys() <= movers2.mover_dict.keys():
            raise SMCViolation(f"ATB with adjoin:"
                               f" {self} not a subset of {movers2}")
        new_mover_dict = movers2.mover_dict  # this is the superset, so start here
        for slot in self.mover_dict:  # go through subset
            if self.mover_dict[slot].multiple:  # these have lists instead of single movers
                if self.mover_dict[slot] != movers2.mover_dict[slot]:
                    # if they're equal, it's already correct. Otherwise, concatenate the lists.
                    new_slot = self.mover_dict[slot]
                    new_slot.combine_multiple_movers(movers2.mover_dict[slot].contents)
                    new_mover_dict[slot] = new_slot
            else:
                if self.mover_dict[slot] != movers2.mover_dict[slot]:
                    raise SMCViolation(f"ATB with merge: {self} != {movers2}")

        return type(self)(new_mover_dict)


# *** Utility functions ***

def atb_combine_slot(m1, m2, mover_type=Movers):
    """
    given two movers with addresses, combine them if their strings are identical
    Return a new TripleItem with span string.addr1.addr2
    @param m1: TripleItem with span of the form string.addr1  (e.g. the.001)
    @param m2: TripleItem with span of the form string.addr2
    @return: TripleItem with span of the form string.addr1.addr2
    """
    log(f"Checking {m1} and {m2}")
    if m1 is None and m2 is None:
        return None
    elif m1 is None or m2 is None:
        raise SMCViolation(f"ATB: only one mover is None")
    elif mover_type == DSMCAddressedMovers:
        # movers need to be interpretted because they need not have the same derivational history
        mover_string = m1.evaluate().collapse()
        added_mover_string = m2.evaluate().collapse()
        # ignore the addresses and compare
        if re.search(r"\w*" + address_sep + "[0,1]*", mover_string) and re.search(r"\w*" + address_sep + "[0,1]*", added_mover_string):
            m1_split = mover_string.split()  # split into [w0.a0, w1.a1, ... ]
            m2_split = added_mover_string.split()
            new_mover = []
            for item_1, item_2 in zip(m1_split, m2_split):
                w1, a1 = item_1.split(address_sep)[0], address_sep.join(item_1.split(address_sep)[1:])
                w2, a2 = item_2.split(address_sep)[0], address_sep.join(item_2.split(address_sep)[1:])
                if w1 == w2:
                    new_mover.append(f"{w1}.{a1}.{a2}")
                else:
                    raise SMCViolation("ATB: movers not the same")
            return Triple(" ".join(new_mover))

        else:
            raise MGError(f"ATB: movers don't have addresses: {m1}, {m2}")
    else:
        raise MGError(f"Not sure")


if __name__ == "__main__":
    A = "A"
    Abar = "Abar"

    who1 = Slot(Abar, contents=Triple("who.01001"))
    print(who1)

    who2 = Slot(Abar, contents=Triple("who.1"))
    print(who2)

    mary = Slot(A, contents=Triple("Mary"), a=True)
    print(mary)

    loc_0 = Slot("loc", contents=Triple("locative.0"), epp=True)
    print(loc_0)
    loc_1 = Slot("loc", contents=Triple("locative.1"), epp=True)

    multi = Slot("E", contents=[Triple("hi")], multiple=True)
    print("multi:", multi)




    my_movers = ListMovers()
    my_movers = my_movers.add_mover(mary)
    my_movers = my_movers.add_mover(multi)
    my_movers = my_movers.add_mover(multi)
    print("my movers", my_movers)

    other_movers = ListMovers()
    other_movers = other_movers.add_mover(who2)
    other_movers = other_movers.add_mover(multi)
    print("other", other_movers)

    other_movers = other_movers.combine_movers(my_movers)
    print("new movers", other_movers)


    # my_movers = my_movers.add_mover(all)
    # print(my_movers)
    # #
    # my_movers = my_movers.add_mover(mary)
    # print(my_movers)
    #
    # m, my_movers = my_movers.remove_mover(Abar)
    # print(m, my_movers)
    #
    # # print(my_movers.epps_filled)
    # # print(my_movers.a_filled)
    #
    # other_movers = Movers()
    # other_movers = other_movers.add_mover(mary)
    # other_movers = other_movers.add_mover(who2)
    # print("other:", other_movers)

    # combo = combine_movers(other_movers, other_movers,
    #                        inner_alg=HMStringTripleAlgebra(), adjoin=False)
    # print(combo)

