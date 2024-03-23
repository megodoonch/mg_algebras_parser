from minimalist_parser.algebras.hm_algebra import HMAlgebra


class SetAlgebra(HMAlgebra):
    def __init__(self):
        super().__init__("set algebra", set)