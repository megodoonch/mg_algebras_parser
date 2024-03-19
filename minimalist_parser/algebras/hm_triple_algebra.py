"""
HM algebra for MGBank

algebra objects are terms over string triples

Not using AL since this is for MGBank

"""
from minimalist_parser.minimalism.prepare_packages.prepare_packages_hm import ATBError
from .hm_algebra import HMAlgebra
from .algebra import AlgebraOp, AlgebraTerm
from .algebra_objects.triples import Triple

# logging
import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# make log() into a logging function (for simplicity)
log = logger.debug


class HMTripleAlgebra(HMAlgebra):
    """
    An algebra over Triples
    Default, these are string triples, but anything iterable should work.
    To use something other than strings, pass, e.g., component_type=list to initialiser.

    Special Attributes:
    - component_type: the type of the three parts of the triple (default str)
    - empty_object is the initialisation of an empty component_type, e.g. str()
    - constant_maker takes a component_type instance, and optionally a label, yielding AlgebraOp(label, instance)
        e.g. constant_maker("cat") = AlgebraOp("cat", Triple("cat"))
        e.g. constant_maker("cat", "cat_label") = AlgebraOp("cat_label", Triple("cat"))
        e.g. constant_maker(["010"], "cat_label") = AlgebraOp("cat_label", Triple(["010"], component_type=list)
    """

    def __init__(self, name=None, component_type=None, addresses=False):
        """
        @param name: the name of the algebra. Default <Component Type> Triple Algebra.
        @param component_type: the type of the three components of the Triple objects. Default is str.
        """
        component_type = str if component_type is None else component_type
        self.meta = {}
        self.meta["component type"] = component_type
        self.meta["addresses"] = addresses
        name = f"{component_type.__name__.title()} Triple Algebra" if name is None else name
        empty_object = AlgebraOp("[empty]", Triple(component_type(), component_type=component_type))
        super().__init__(name=name, domain_type=Triple, zero=empty_object, meta=self.meta)
        # self.component_type = component_type
        self.add_constant_maker(self.default_constant_maker)
        # print(self.meta)

    def __repr__(self):
        return self.name

    def make_leaf(self, name, function=None):
        """
        Makes a new item with given string as head
        @param function:
        @param name: str
        @return: AlgebraTerm
        """
        # turn h into < ,h, >
        if function is None:
            function = self.domain_type(name, component_type=self.meta["component type"])
        return AlgebraTerm(AlgebraOp(name, function))

    def default_constant_maker(self, word, label=None):
        label = label if label is not None else str(word)
        return AlgebraOp(label, self.domain_type(word, component_type=self.meta["component type"]))

    def concat_left(self, args: list[Triple]):
        """
        Concatenate args[1] to the left of args[0]
        @param args: list of two Triples.
        @return: Triple
        """
        assert len(args) == 2, "Concat requires exactly 2 arguments"
        assert issubclass(type(args[0]), Triple), f"concat requires Triples, but we have {type(args[0])}"
        assert issubclass(type(args[1]), Triple), f"concat requires Triples, but we have {type(args[1])}"
        functor, other = args[0], args[1]
        # Use a concat method of the algebra object
        return functor.concat_left(other)

    def suffix(self, args: list[Triple]):
        assert len(args) == 2, "suffix requires exactly 2 arguments"
        assert issubclass(type(args[0]), Triple) and issubclass(type(args[1]), Triple), f"suffix requires Triples," \
                                                                    f" but got {type(args[0])} and {type(args[1])}"
        # arg0 must be lexical (here, just a head)
        assert not args[1].left and not args[1].right
        # Use a concat method of the algebra object
        return args[0].concat_suffix(args[1].head)

    def prefix(self, args: list[Triple]):
        assert len(args) == 2, "prefix requires exactly 2 arguments"
        assert issubclass(type(args[0]), Triple) and issubclass(type(args[1]), Triple), f"prefix requires Triples," \
                                                                    f" but got {args[0]} ({type(args[0]).__name__}) and {args[1]} ({type(args[1]).__name__})"
        assert not args[1].left and not args[1].right
        # Use a concat method of the algebra object
        return args[0].concat_prefix(args[1].head)


def combine_leaf_addresses(term_1: AlgebraTerm, term_2: AlgebraTerm):
    if term_1.is_leaf() and term_2.is_leaf():
        term_1_addresses = term_1.evaluate().head
        term_2_addresses = term_2.evaluate().head
        logger.debug(f"combining addresses: {term_1_addresses}, {term_2_addresses}")
        new_addresses = [term_1_address + term_2_address for term_1_address, term_2_address in zip(term_1_addresses, term_2_addresses)]
        logger.debug(f"new addresses: {new_addresses}")
        term_1.parent.function = Triple(new_addresses)
        logger.debug(f"new function: {term_1.parent.function}")
        return term_1, term_2
    elif term_1.is_leaf() or term_2.is_leaf():
        raise ATBError(f"Cannot combine {term_1} and {term_2} with ATB phrasal movement")
    elif len(term_1.children) != len(term_2.children):
        raise ATBError(f"Cannot combine {term_1} and {term_2} with ATB phrasal movement")
    else:
        new_children = [combine_leaf_addresses(child_1, child_2) for child_1, child_2 in zip(term_1.children, term_2.children)]
        term_1.children = [pair[0] for pair in new_children]
        return term_1, term_2



if __name__ == "__main__":
    b = HMTripleAlgebra()
    from minimalist_parser.minimalism.prepare_packages.triple_prepare_package import HMTriplesPreparePackages

    print(b)

    the = b.make_leaf("the")
    #    print(the.can_hm())

    past = b.make_leaf("t")

    cat = b.make_leaf("cat")
    print(cat)

    slept = b.make_leaf("slep")
    print(slept)

    conjunct = b.make_leaf("and")
    dog = b.make_leaf("dog")
    print(dog)

    the_cat = b.concat_right([the.parent.function, cat.parent.function])
    print(the_cat)

    the_cat_slept = b.concat_left([slept.parent.function, the_cat])

    print(the_cat_slept)

    t = AlgebraTerm(b.concat_left_op, [slept, AlgebraTerm(b.concat_right_op, [the, cat])])

    print(t.evaluate())

    prep = HMTriplesPreparePackages()

    prepared = prep.prefix(past, t)
    # print(prepared)

    print(prepared[0])
    print(prepared[0].evaluate())
    print()
    print(prepared[1])
    print(prepared[1].evaluate())

    t = AlgebraTerm(b.concat_left_op, [prepared[0], prepared[1]])
    print(t.evaluate())
