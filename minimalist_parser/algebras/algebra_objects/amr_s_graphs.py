"""
An attempt to fix coordination
"""
from minimalist_parser.algebras.algebra import AlgebraError
from minimalist_parser.algebras.algebra_objects.graphs import SGraph, apply, modify

import logging
VERBOSE = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log = logger.debug


class AMRSGraph(SGraph):
    def __init__(self, nodes: set[int] = None, edges=None, node_labels=None, sources=None, root: int = None):
        super().__init__(nodes, edges, node_labels, sources, root)
        self.OP = "OP"

    def add_op_source(self):
        """
        Add an opi edge from the root. Its target is an OP source,
            and the i is the next up from whatever is already there.
        @return: None, modify in place
        """
        if not self.root_is_conjunction():
            raise AlgebraError("Can't add conjuncts to a non-conjunction")
        # see what opis we alraedy have
        opis = []
        for target, label in self.edges[self.root]:
            i = self.ith_conjunct(label)
            if i is not None:
                opis.append(i)
        logger.debug(opis)
        next_opi = max(opis) + 1
        # add an edge to a new OP source, root -opi->OP
        next_node = max(self.nodes) + 1
        self.nodes.add(next_node)
        self.edges[self.root].append((next_node, f"op{next_opi}"))
        self.add_source(next_node, self.OP)

    @staticmethod
    def ith_conjunct(label):
        """
        Get the next possible opi given what opis are already edges from the root.
        e.g. if the root has an op1 and an op2, return 3.
        @param label: str
        @return: int
        """
        if label.startswith("op"):
            try:
                i = int(label[2:])
                return i
            except Exception as e:
                print(e)
                return
        return

    def root_is_conjunction(self):
        """
        In case we want to constrain this
        @return: bool
        """
        return True

    def adjunctize(self, edge_label="mod", mod_source="M"):
        """
        TODO is this a prepare function?
        Make any graph into a modifier by adding an incoming edge to the root
            from a mod_source
        @param edge_label: the label to give the new edge (default mod)
        @param mod_source: the source to add (default M)
        @return: modify in place
        """
        assert mod_source not in self.sources
        new_node = max(self.nodes) + 1
        self.nodes.add(new_node)
        self.sources[mod_source] = new_node
        self.edges[new_node] = [(self.root, edge_label)]




if __name__ == "__main__":

    like = AMRSGraph(
        {0, 1, 2},
        {
            1: [(0, "ARG0"), (2, "ARG1")]
        },
        {1: "like"},
        {"S": 0, "O": 2},
        1
    )

    tried = AMRSGraph(
        {0, 1, 2},
        {
            1: [(0, "ARG0"), (2, "ARG1")]
        },
        {1: "try"},
        {"S": 0, "O": 2},
        1
    )

    mary = AMRSGraph(
        {0, 1},
        edges={0: [(1, "name")]},
        node_labels={0: "person", 1: "Mary"},
        root=0
    )

    cat = AMRSGraph(
        {0},
        node_labels={0: "cat"},
        root=0
    )

    and_graph = AMRSGraph(
        {0, 1},
        {
            1: [(0, "op1")]
        },
        {1: "and"},
        {"OP": 0},
        1
    )


    and_cat = apply("OP", and_graph, cat)
    print(and_cat)

    and_cat.add_op_source()

    mary_and_cat = apply("OP", and_cat, mary)
    print(mary_and_cat)