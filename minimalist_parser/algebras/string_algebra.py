from minimalist_parser.algebras.algebra import *
from minimalist_parser.algebras.hm_algebra import HMAlgebra


def concatenate(s, joiner=" "):
    """for nicer concatenation"""
    return joiner.join([w for w in s if len(w) > 0])


# some empty categories
trace = AlgebraOp("t", "")  # a string algebra object interpreted as silent
epsilon = AlgebraOp("<e>", "")


class BareTreeStringAlgebra(HMAlgebra):
    """
    A string algebra in which the concatenate operation names are > and <, and for head movement, <_h and >_h.
    This makes for a simple string algebra in which the algebra terms are Bare Trees a la Stabler 1997.

    Note that this is different from a tree-building algebra, in which the objects built are actually trees.
    """

    def __init__(self, name=None, zero=None, ops=None, syntax_op_names=None, meta=None):
        """
        Initialise a string-building algebra with Bare Tree algebra terms.
        @param name: str: Algebra name, default "String Algebra".
        @param zero: The empty AlgebraOp, default <e> with interpretation "".
        @param ops: dict of operation name to AlgebraOp. Default < and > both interpreted as self.concat_right,
                                                            and <_h and >_h, both interpreted as self.suffix.
        @param syntax_op_names: set[str]: Which operation names count a syntax, as opposed fto morphology. Default <,>.
                                            Used to find the head of the term.
        @param meta: dict, any extra info you want to add.
        """
        syntax_op_names = {"<", ">"} if syntax_op_names is None else syntax_op_names
        zero = epsilon if zero is None else zero
        meta = meta if meta is not None else {"component type": str}
        name = name if name is not None else "String Algebra"
        super().__init__(name, domain_type=str, zero=zero, ops=ops, syntax_op_names=syntax_op_names, meta=meta)
        if ops is None:
            self.ops["concat_right"] = AlgebraOp("<", self.concat_right)
            self.ops["concat_left"] = AlgebraOp(">", self.concat_right)
            self.ops["suffix"] = AlgebraOp("<_h", self.suffix)
            self.ops["prefix"] = AlgebraOp(">_h", self.suffix)
        self.trace = AlgebraTerm(trace)  # trace marker for moved objects; <t> interpreted as ""
        self.add_constant_maker()

    def concat_right(self, args: list[str]):
        """
        Concatenate the args with a space.
        @param args: list[str]
        @return: str
        """
        return concatenate(args)

    def suffix(self, args):
        """
        Concatenate the args with a space.
        This is kept separate from concat_right in case we want to concatenate with, say, "-" for head movement.
        @param args: list[str]
        @return: str
        """
        return concatenate(args)


# *** String Algebra ***

# note here heads are always on the right.
concat_op = AlgebraOp("*", lambda s: concatenate(s))
inv_concat_op = AlgebraOp("*^-1", lambda s: concatenate(rev(s)))
suffix = AlgebraOp("-", lambda s: concatenate(s, "-"))
prefix = AlgebraOp("-^-1", lambda s: concatenate(rev(s), "-"))

string_alg = HMAlgebra(name="String Algebra",
                       domain_type=str,
                       zero=epsilon,
                       ops={"concat_right": concat_op,
                            "concat_left": inv_concat_op,
                            "suffix": suffix,
                            "prefix": prefix
                            },
                       syntax_op_names={"*", "*^-1"}
                       )

string_alg.add_constant_maker()
