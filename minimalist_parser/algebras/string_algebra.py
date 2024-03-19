from minimalist_parser.algebras.algebra import *
from minimalist_parser.algebras.hm_algebra import HMAlgebra


def concatenate(s, joiner=" "):
    """for nicer concatenation"""
    return joiner.join([w for w in s if len(w) > 0])

# *** String Algebra ***


concat_op = AlgebraOp("<", lambda s: concatenate(s))
inv_concat_op = AlgebraOp(">", lambda s: concatenate(rev(s)))
suffix = AlgebraOp("<_h", lambda s: concatenate(s, "-"))
prefix = AlgebraOp(">_h", lambda s: concatenate(rev(s), "-"))


trace = AlgebraOp("t", "")  # a string algebra object interpreted as silent
epsilon = AlgebraOp("<e>", "")

string_alg = HMAlgebra(name="String Algebra",
                       domain_type=str,
                       zero=epsilon,
                       ops={"concat_right": concat_op,
                          "concat_left": inv_concat_op,
                          "suffix": suffix,
                          "prefix": prefix
                          },
                       syntax_op_names={"<", ">"}
                       )
#                          , zero=epsilon.function, name="String Algebra")
string_alg.add_constant_maker()

class BareTreeStringAlgebra(HMAlgebra):
    def __init__(self, name=None, zero=None, ops=None, syntax_op_names=None, meta=None):
        syntax_op_names = {"<", ">"} if syntax_op_names is None else syntax_op_names
        zero = epsilon if zero is None else zero
        meta = meta if meta is not None else {"component type": str}
        super().__init__(name, domain_type=str, zero=zero, ops=ops, syntax_op_names=syntax_op_names, meta=meta)
        if ops is None:
            self.ops["concat_right"] = AlgebraOp("<", self.concat_right)
            self.ops["concat_left"] = AlgebraOp(">", self.concat_right)
            self.ops["suffix"] = AlgebraOp("<_h", self.suffix)
            self.ops["prefix"] = AlgebraOp(">_h", self.suffix)
        self.trace = AlgebraTerm(trace)

    def concat_right(self, args):
        return concatenate(args)

    def suffix(self, args):
        return concatenate(args)