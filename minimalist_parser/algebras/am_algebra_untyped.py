"""
implements a typeless version of the AM algebra. We only care about whether we can apply the operation, not whether we should
"""
import logging
from copy import deepcopy

from overrides import overrides

from .algebra import *
from .algebra_objects.amr_s_graphs import AMRSGraph
from .algebra_objects.graphs import SGraph, apply, modify


class AMAlgebra(Algebra):
    """
    A typeless AM algebra.
    Apply and Modify work as long as the underlying operations work, otherwise they throw the underlying errors.
    Domain: SGraphs.
    Binary operations: Apply and Modify (by source).
    Attributes:
        zero: empty graph.
        domain_type: the type SGraph.
        sources: list of strings (could be anything) for the source of the SGraphs.
        app_only: sources that are only used in APP.
        mod_only: sources that are only used in MOD.
        ops: dict of operation names to AlgebraOps. Traditionally just the non-constants
        constant_maker: a function that makes constant AlgebraOps.
         For SGraphs you probably want to add this after initialisation by adding a dict of constants,
         which makes the constant maker a look-up function.

    """

    def __init__(self, sources=None, app_only=None, mod_only=None, graph_type=SGraph):
        super().__init__(name="AM algebra", domain_type=graph_type)
        self.add_constant_maker()
        print("added constant maker")

        self.OP = "OP"

        # some AMR defaults
        if sources is None:
            sources = ["M", "S", "O"]
            for i in range(1, 8):
                sources.append(f"{self.OP}{i}")
            if self.domain_type == AMRSGraph:
                sources.append(self.OP)
            else:
                for i in range(1, 20):
                    sources.append(f"{self.OP}{i}")
            if app_only is None:
                app_only = [source for source in sources if source.startswith(self.OP)]
            if mod_only is None:
                mod_only = ["M"]

        self.sources = sources
        self.app_only = app_only
        self.mod_only = mod_only

        self.ops = {}

        #print(self.sources)
        #print(self.mod_only)

        # all Algebra operations defined here
        for s in [x for x in self.sources if x not in self.mod_only]:
            function = self.make_apply_operation(s)
            name = f"App_{s}"
            op = AlgebraOp(name, function)
            self.ops[op.name] = op
        for s in [x for x in self.sources if x not in self.app_only]:
            function = self.make_modify_operation(s)
            name = f"Mod_{s}"
            op = AlgebraOp(name, function)
            self.ops[op.name] = op
        if self.domain_type == AMRSGraph:
            name = "Add_conjunct"
            self.ops[name] = AlgebraOp(name, self.add_conjunct)

    def __repr__(self):
        return self.name

    def ops_repr(self):
        apps = []
        mods = []
        other = []
        for op in self.ops:
            if op[:3].lower() == "app":
                apps.append(op)
            elif op[:3].lower() == "mod":
                mods.append(op)
            else:
                other.append(op)
        ret = "\n\tApp:\n\t\t"
        for op in sorted(apps):
            ret += f" {op}"
        ret += "\n\tMod:\n\t\t"
        for op in sorted(mods):
            ret += f" {op}"
        ret += "\n\tOther:\n\t\t"
        for op in sorted(other):
            ret += f" {op}"
        return ret

    @staticmethod
    def make_apply_operation(source):
        def apply_to_list(args):
            head = args[0]
            argument = args[1]
            # def apply_source(head: SGraph, argument: SGraph):
            return apply(source, head, argument)
            # return apply_source

        return apply_to_list

    @staticmethod
    def make_modify_operation(source):
        def modify_source(args):
            head = args[0]
            modifier = args[1]
            return modify(source, head, modifier)

        return modify_source

    def make_leaf(self, name, function=None):
        return AlgebraTerm(self.constant_maker(name))

    @overrides
    def default_constant_maker(self, word, label=None):
        """
        Makes unary graph with word as label
        """
        logger.debug("using default constant maker")
        return AlgebraOp(word, self.domain_type({0}, node_labels={0: word}, root=0))

    def add_conjunct(self, args):
        head, conjunct = args[0], args[1]
        try:
            head.add_op_source()
            return apply(self.domain_type().OP, head, conjunct)
        except AttributeError:
            raise NotImplementedError

    def adjunctize(self, graph: SGraph, edge_label="mod", mod_source="M"):
        """
        Make any graph into a modifier by adding an incoming edge to the root
            from a mod_source
        @param edge_label: the label to give the new edge (default mod)
        @param mod_source: the source to add (default M)
        @return: SGraph
        """
        assert mod_source not in graph.sources
        new_self = deepcopy(graph)
        new_node = max(new_self.nodes) + 1
        new_self.nodes.add(new_node)
        new_self.sources[mod_source] = new_node
        new_self.edges[new_node] = [(new_self.root, edge_label)]
        return new_self

    def make_adjunctize_operation(self, edge_label, source):
        def adjunctize(modifier):

            return self.adjunctize(modifier, edge_label, source)

        return adjunctize


# if __name__ == "__main__":
like = AMRSGraph(
    {0, 1, 2},
    {
        1: [(0, "ARG0"), (2, "ARG1")]
    },
    {1: "like"},
    {"S": 0, "O": 2},
    1
)

slept = AMRSGraph(
    {0, 1},
    {
        1: [(0, "ARG0")]
    },
    {1: "sleep"},
    {"S": 0},
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

# cat = AMRSGraph(
#     {0},
#     node_labels={0: "cat"},
#     root=0
# )

whistling = AMRSGraph(
    {0, 1, 2},
    {
        1: [(2, "ARG0")],
        0: [(1, "mod")]
    },
    {1: "whistle"},
    {"S": 2, "M": 0},
    1
)

and_graph = AMRSGraph(
    {0, 1},
    {
        1: [(0, "op1")]
    },
    {1: "and"},
    {AMRSGraph().OP: 0},
    1
)

# k = g + h
# print(k)

m = AMRSGraph(
    {0, 1, 2},
    {1: [(2, "with"), (0, "mod")]},
    {0: "cute", 2: "glee"},
    sources={"M": 1},
    root=0
)


def make_predicate(label, arg_numbers):
    """
    Make a default sort of predicate in which only the root is labelled and all argument source are promoted.
    e.g. if arg_numbers = [0,2] edges will be ARG0 and ARG2 but sources will be S, O.
    @param label: root node label
    @param arg_numbers: list of integers for edge labels ARG{i}
    @return: AMRSGraph
    """
    edge_labels = sorted([f"ARG{n}" for n in arg_numbers])
    nodes = range(len(arg_numbers))
    sources = {"S": nodes[0]}
    if len(arg_numbers) > 1:
        sources["O"] = nodes[1]
        if len(arg_numbers) > 2:
            for i, node in enumerate(nodes[2:]):
                sources[f"O{i}"] = node
    root = len(arg_numbers)
    edges = {root: []}
    for l,t in zip(edge_labels, nodes):
        edges[root].append((t, l))
    nodes = set(nodes)
    nodes.add(root)
    ret = AMRSGraph(root=root, edges=edges, sources=sources, nodes=nodes, node_labels={root: label})
    logging.debug(f"Built predicate {ret}")
    return ret


# p = k + m
# print(p)
# print(h)
sleep = make_predicate("sleep-01", [0])
dream = make_predicate("dream-01", [0])

vocabulary = {"mary": mary,
              "tried": tried,
              "like": like,
              "cute": m,
              "whistling": whistling,
              "and": and_graph,
              "sleep": sleep,
              "slept": sleep,
              "dream": dream,
              "dreamt": dream,
              }

am_alg = AMAlgebra(graph_type=AMRSGraph)
am_alg.add_constants({name: AlgebraOp(name, vocabulary[name]) for name in vocabulary})


# print("VOCAB:")
# for constant in am_alg.constants:
#     print(constant, am_alg.constants[constant].function)



    # print(am_alg.constants)

    # print(am_alg.sources)
    # print(am_alg.ops)
    #
    # g = am_alg.ops["App_O"].function([am_alg.constant_maker("like").function, am_alg.constant_maker("cat").function])
    #
    # print("Output\n", g)

    # t1 = AlgebraTerm(am_alg.ops["App_O"], [am_alg.make_leaf("like"), am_alg.make_leaf("cat")])
    # t2 = AlgebraTerm(am_alg.ops["App_O"], [am_alg.make_leaf("tried"), t1])
    # t3 = AlgebraTerm(am_alg.ops["App_S"], [t2, am_alg.make_leaf("Meaghan")])
    #
    # t1 = AlgebraTerm(am_alg.ops["App_OP"], [am_alg.make_leaf("and"), am_alg.make_leaf("cat")])
    # print(t1.evaluate())
    #
    # t2 = AlgebraTerm(am_alg.ops["Add_conjunct"], [t1, am_alg.make_leaf("Meaghan")])
    # print(t2.evaluate())
    #
    # t2 = AlgebraTerm(am_alg.ops["Add_conjunct"], [t2, am_alg.make_leaf("Tunya")])
    # print(t2.evaluate())
    #
    # print(am_alg)