from minimalist_parser.algebras.algebra import AlgebraTerm, AlgebraError
from minimalist_parser.minimalism.prepare_packages.prepare_packages_hm import PreparePackagesHM


class PreparePackagesBareTrees(PreparePackagesHM):
    """
    Prepare Packages for string algebra with bare trees as terms
    """
    def __init__(self, name=None, inner_algebra=None):
        if inner_algebra is None:
            from minimalist_parser.algebras.string_algebra import BareTreeStringAlgebra
            inner_algebra = BareTreeStringAlgebra()
        if name is None:
            name = f"{inner_algebra} prepare"
        super().__init__(name, inner_algebra)

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
        new_functor = self.add_affix(argument_head, functor, op)

        return new_functor, updated_argument

    def remove_head(self, term: AlgebraTerm):
        """
        Finds head by looking down the left branches until we hit a leaf or a non-concat function
        Returns the term an empty item in place of the head subterm
        @param term: the AlgebraTerm to update
        @return: the updated AlgebraTerm paired with the extracted head
        """
        if self.is_head_subterm(term):
            # if we're at the head, return a trace if possible, otherwise an empty leaf
            try:
                return self.inner_algebra.trace
            except AttributeError:
                ret = self.inner_algebra.empty_leaf
                assert ret is not None, f"{self.inner_algebra} has no trace"
                return ret
        elif len(term.children) > 1:
            if term.parent.name == "<":
                # the head is on the left
                remaining_children = term.children[1:]
                new_head = self.remove_head(term.children[0])
                return AlgebraTerm(term.parent, [new_head] + remaining_children)
            elif term.parent.name == ">":
                # the head is on the right
                remaining_children = term.children[:-1]
                new_head = self.remove_head(term.children[1])
                return AlgebraTerm(term.parent, remaining_children + [new_head])
            else:
                raise AlgebraError(f"Node {term.parent.name} is neither <, >, nor a head.")
        else:
            return AlgebraTerm(term.parent, [])

    def get_head(self, term: AlgebraTerm):
        """
        Finds head by looking down the left branches until we hit something other than < or >.
        @param term: the AlgebraTerm we want the head of.
        @return: AlgebraTerm: the head of the input term.
        """
        if term.parent.name == "<":
            return self.get_head(term.children[0])
        elif term.parent.name == ">":
            return self.get_head(term.children[1])
        else:
            return term
