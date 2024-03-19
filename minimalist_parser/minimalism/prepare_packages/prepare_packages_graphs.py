from abc import ABC

from minimalist_parser.algebras.algebra import Algebra, AlgebraTerm, AlgebraOp
from minimalist_parser.algebras.algebra_objects.graphs import SGraph
from minimalist_parser.algebras.am_algebra_untyped import AMAlgebra
from minimalist_parser.minimalism.prepare_packages.prepare_packages import PreparePackages


def make_prepare_function(op, edge_label=None, source=None, inner_algebra=AMAlgebra):
    function = op
    if op == inner_algebra.add_conjunct:
        name = "add_conjunct"
    elif op == inner_algebra.make_adjunctize_operation:
        name = f"adjunctize_{source}_{edge_label}"
        function = op(edge_label, source)
    else:
        name = "prepare"

    return AlgebraOp(name, function)


class PreparePackagesGraph(PreparePackages):
    """
    # TODO test
    """
    def __init__(self, name, inner_algebra: AMAlgebra):
        super().__init__(name, inner_algebra)

    def __repr__(self):
        return self.name

    def adjunctize(self, head: AlgebraTerm, modifier: AlgebraTerm):
        """
        Prepare Package
        give modifier a mod edge with M source
        @param head: head Term
        @param modifier: modifier Term
        @return: head term, updated modifier term
        """
        new_modifier = AlgebraTerm(make_prepare_function(self.inner_algebra.make_adjunctize_operation), [modifier])
        return head, new_modifier

    def add_conjunct(self, head: AlgebraTerm, argument: AlgebraTerm):
        new_head = AlgebraTerm(make_prepare_function(self.inner_algebra.add_conjunct), [head])
        return new_head, argument




