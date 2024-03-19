from ...algebras.algebra import AlgebraTerm
from ...algebras.hm_triple_algebra import HMTripleAlgebra
from .prepare_packages_hm import PreparePackagesHM
import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
log = logger.debug


class HMAddressedTriplesPreparePackages(PreparePackagesHM):
    """
    Defines pairs of tree homomorphisms for Triples of strings for head movement
    These update the functor and argument, for instance by moving a head
    """

    def __init__(self):
        super().__init__("HM Address Triples Prepare Packages", HMTripleAlgebra(component_type=list))

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
        @param other: AlgebraTerm: the selectee.
        @param functor: AlgebraTerm: the selector
        @return: pair of AlgebraTerms with: the head, other_rest + functor_rest
                                i.e. (_, h, _) and
                                (other.left + other.right + functor.left
                                 + functor.right)
        """
        updated_other, other_head = self.extract_head(other)
        updated_functor, functor_head = self.extract_head(functor)

        # We assume these are string-identical in the string triple interpretation,
        # and just combine the two source addresses
        functor_head_addresses = functor_head.evaluate().head
        other_head_addresses = other_head.evaluate().head
        logger.debug(f"applying hm_atb prepare package to {functor_head_addresses} and {other_head_addresses}")
        # these are heads, but because of HM they could be complex.
        # There should be a pointwise correspondence, so we concatenate each list within the list.
        new_head_addresses = [functor_head_address + other_head_address
                              for functor_head_address, other_head_address
                              in zip(functor_head_addresses, other_head_addresses)]
        rest = AlgebraTerm(self.inner_algebra.ops["concat_right"], [updated_other, updated_functor])

        return (self.inner_algebra.make_leaf(name=functor_head.parent.name,
                                             function=self.inner_algebra.domain_type(new_head_addresses)),
                rest)
