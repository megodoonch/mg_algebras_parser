"""
Base class for algebras with head movement
"""
import logging
from abc import ABC

from overrides import overrides

from .algebra import AlgebraTerm, AlgebraOp, Algebra, AlgebraError
from ..trees.trees import Tree

# logging: change VERBOSE to False to stop printing everything
VERBOSE = False


class HMError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message=None):
        self.message = message


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

log = logger.debug



class HMAlgebra(Algebra, ABC):
    """
    Parent class for HM algebras compatable with MGBank
    Most methods need to be implemented for the specific algebra
    Attributes:
        name: str: name for the algebra
        domain_type: Type: the type of the domain of the algebra, e.g. Triple, IntervalItem
        ops: dict str (op name) : AlgebraOp. Optional. Default
            {'concat_right': AlgebraOp('concat_right', self.concat_right)
             'concat_left': AlgebraOp('concat_left', self.concat_left),
             'suffix': AlgebraOp('suffix', self.suffix),
             'prefix': AlgebraOp('prefix', self.prefix)
             }
        syntax_op_names: list of str: these operations are syntactic, as opposed to morphological,
                                        and so for the purposes of finding the head of a term,
                                        we stop when we find any operation not in this list.
    """

    def __init__(self, name, domain_type, zero=None, meta=None, ops=None, syntax_op_names=None):
        """
        An HMAlgebra has a name and specifies the type of its inner items
        @param name: str
        @param domain_type: type, e.g. str or Tree or PairItem
        @param zero: and AlgebraOp to use for default empty heads (optional).
                    Default: AlgebraOp('<e>', domain_type())
        """

        super().__init__(name=name, domain_type=domain_type, zero=zero, meta=meta, ops=ops)

        # all Algebra operations defined here
        if ops is None:
            concat_left_op = AlgebraOp("concat_left", self.concat_left)
            concat_right_op = AlgebraOp("concat_right", self.concat_right)
            suffix_op = AlgebraOp("suffix", self.suffix)
            prefix_op = AlgebraOp("prefix", self.prefix)

            for op in [concat_right_op, concat_left_op, suffix_op, prefix_op]:
                self.ops[op.name] = op

        # as opposed to morphology
        self.syntax_op_names = ["concat_left", "concat_right"] if syntax_op_names is None else syntax_op_names

    def __repr__(self):
        return f"{self.name}"  # over {self.domain_type.__name__}s"

    # @overrides
    # def make_leaf(self, head, function=None):
    #     """
    #     Makes a new item with given string as head
    #     @param function: use this as the function if given, otherwise make it automatically
    #     @param head: str: name of leaf
    #     @return: AlgebraTerm
    #     """
    #     # turn h into < ,h, >
    #     if function is not None:
    #         return AlgebraTerm(AlgebraOp(head, function))
    #     else:
    #         leaf = self.domain_type(head)
    #         return self.object2leaf(leaf)
    #
    # def object2leaf(self, inner_object):
    #     """
    #     Makes an AlgebraTerm with already-built inner object
    #     The name of the leaf is the repr of the item given
    #     @param inner_object: e.g. Triple
    #     @return: AlgebraTerm
    #     """
    #     return AlgebraTerm(AlgebraOp(repr(inner_object), inner_object))

    # *** algebra operations ****

    def concat_left(self, args):
        """
        Concatenate args[1] to the left of args[0]
        If functor is lexical, we can leave a gap, since HM can apply later
        Otherwise, we concatenate to the rest, if possible, or concat all 3
        Raises errors when impossible
        @param args: list of two algebra objects
        @return: algebra object
        """
        assert len(args) == 2, "Concat requires exactly 2 arguments"
        assert (isinstance(args[0], self.domain_type)
                and isinstance(args[1], self.domain_type)
                ), f"args are of type {type(args[0])} and {type(args[1])} but should be {self.domain_type}"
        functor, other = args[0], args[1]
        return other + functor

    def concat_right(self, args):
        """
        Concatenate args[1] to the right of args[0]
        @param args: list of two algebra objects
        @return: algebra object
        """
        assert len(args) == 2, "Concat requires exactly 2 arguments"
        assert type(args[0]) == self.domain_type, f"require types {self.domain_type} but got {type(args[0])}: {args[0]}"
        functor, other = args[0], args[1]
        # logging.debug(f"concat_right of {functor} and {other} with component types {functor.component_type} and {other.component_type}")
        return functor + other

    def suffix(self, args: list):
        """
        2 algebra objects (?) to one algebra object (?)
        @param args: list of length 2
        @return:
        """
        assert len(args) == 2, "suffix requires exactly 2 arguments"
        return args[0] + args[1]

    def prefix(self, args):
        assert len(args) == 2, "prefix requires exactly 2 arguments"
        return args[1] + args[0]

    def label2algebra_op(self, label: str):
        """
        Creates an algebra op from the name of the method
        @param label: str: must be the name of a method in this class
        @return: AlgebraOp with that name and function
        """
        try:
            f = getattr(self, label)
            return AlgebraOp(label, f)
        except AttributeError:
            raise AlgebraError(f"No such method as {label}")

    def tree2term(self, tree: Tree):
        """
        Turns a Tree with nodes labelled
        @param tree: Tree (as defined in trees.py)
        @return: AlgebraTerm
        """
        if tree.children is None:
            return self.make_leaf(tree.parent)
        else:
            kids = [self.tree2term(kid) for kid in tree.children]
            return AlgebraTerm(self.label2algebra_op(tree.parent), kids)


if __name__ == "__main__":
    pass
