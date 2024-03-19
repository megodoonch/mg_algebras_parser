import re
from abc import ABC

from ...algebras.algebra import AlgebraTerm
from ...algebras.hm_triple_algebra import HMTripleAlgebra
from .prepare_packages_hm import PreparePackagesHM, ATBError


class HMTriplesPreparePackages(PreparePackagesHM):
    """
    Defines pairs of tree homomorphisms for Triples of strings for head movement
    These update the functor and argument, for instance by moving a head
    """

    def __init__(self, name=None, inner_algebra=None):
        if name is None:
            name = "HM String Triples Prepare Packages"
        if inner_algebra is None:
            inner_algebra = HMTripleAlgebra()
        super().__init__(name, inner_algebra)

