import logging
import sys
from copy import deepcopy

from minimalist_parser.algebras.algebra import AlgebraError

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class GraphError(Exception):
    def __init__(self, message=None):
        self.message = message


class SetError(Exception):
    def __init__(self, message=None):
        self.message = message


def merge_dicts(d1, d2):
    """
    Merge dictionaries. Allow conflicts if entry in one is None (use the other), otherwise raise error if conflicts
    @param d1: dict
    @param d2: dict
    @return: dict
    """
    new_d1 = {key: value for key, value in d1.items() if value is not None}
    new_d2 = {key: value for key, value in d2.items() if value is not None}
    for key in new_d1:
        if key in new_d2 and new_d1[key] != new_d2[key]:
            if type(new_d1[key] == list):
                # put the updated list in d1
                new_d1[key] += new_d2[key]
                # print("new d1", new_d1)
            else:
                raise SetError("Conflicting dictionary entries")
    # print("merge_dict", new_d1, new_d2)
    # d1 wins
    new_d2.update(new_d1)
    return new_d2  # {**new_d1, **new_d2}


class SGraph:
    """
    s-graphs a la Courcelle et al.
    Nodes can be annotated with extra labels called "sources", which can be used to target particular nodes for operations.
    Attributes:
        nodes: set of ints
        edges: dict from nodes to lists of (target, edge label)
        node_labels: dict from node to label
        sources: dict from source to node
        root: int: specially marked node
    """

    def __init__(self, nodes: set[int] = None, edges=None, node_labels=None, sources=None, root: int = None):
        self.nodes = set() if nodes is None else nodes
        self.edges = {} if edges is None else edges
        self.node_labels = {} if node_labels is None else node_labels
        self.sources = {} if sources is None else sources
        self.root = root

    def get_source_for_node(self, n):
        sources = []
        for source in self.sources:
            if n == self.sources[source]:
                sources.append(source)
        return sources

    def __str__(self):
        # ret = ""
        # nodes = [node for node in self.nodes if not self.is_root(node)]
        # nodes = [self.root] + nodes
        # for n in nodes:
        #     label = self.get_node_label(n)
        #     sources = self.get_source_for_node(n)
        #     if self.root == n:
        #         sources.append("rt")
        #     s = ""
        #     if label is not None:
        #         s += label
        #     for source in sources:
        #         s += f"<{source}>"
        #     ret += f"{n}/{s}\n"
        #     if n in self.edges:
        #         for target, label in self.edges[n]:
        #             ret += f"\t:{label} {target}\n"

        return repr(self)

    def __repr__(self):
        ret = f""
        ret += f"rt:\t\t\t{self.root}\n"
        ret += f"nodes:\t\t{self.nodes}\n"
        ret += f"labels:\t\t{self.node_labels}\n"
        ret += f"sources:\t{self.sources}\n"
        ret += f"edges:\n"
        for n in self.edges:
            for target, label in self.edges[n]:
                ret += f"\t {n} {label} {target}\n"
        return ret

    def is_root(self, node):
        return node == self.root

    def get_node_label(self, n):
        return self.node_labels.get(n, "")

    def add_node_label(self, n, label):
        if n in self.node_labels:
            raise GraphError(f"Nodes can't have two labels ({n} is already labelled {self.node_labels[n]})")
        self.node_labels[n] = label

    def replace_node(self, old_node: int, new_node: int):
        """
        replace a node in place everywhere it appears.
        @param old_node: int.
        @param new_node: int.
        """
        assert new_node not in self.nodes, f"can't replace node {old_node} with {new_node}: it's already present"
        logger.debug(f"replacing {old_node} with {new_node} in \n {self}")
        # root
        if self.root == old_node:
            self.root = new_node
            logger.debug(f"updated root yielding \n{self}")

        # nodes
        self.nodes.remove(old_node)
        self.nodes.add(new_node)
        logger.debug(f"updated nodes to \n{self.nodes}")

        # edges
        if old_node in self.edges:
            # rename the node qua edge source
            edges = self.edges.pop(old_node)
            self.edges[new_node] = edges
        # rename the node qua edge target
        for source in self.edges:
            new_edges = []
            for target, label in self.edges[source]:
                if target == old_node:
                    new_edges.append((new_node, label))
                else:
                    new_edges.append((target, label))
            self.edges[source] = new_edges

        # sources
        for source in self.sources:
            if self.sources[source] == old_node:
                self.sources[source] = new_node

        # node labels
        if old_node in self.node_labels:
            label = self.node_labels.pop(old_node)
            self.node_labels[new_node] = label

    def __add__(self, other):
        """
        Add two Sgraphs together by merging their common source
         and otherwise renaming the nodes of other to keep them separate.
        """
        assert type(other) == type(self), f"{self} type {type(self)}, {other} type {type(other)}"
        new_other = deepcopy(other)
        new_self = deepcopy(self)
        # rename all sources in other to match self
        keep = []
        assert type(other.sources) == dict, f"{other.sources} is not a dict"
        logger.debug(f"changing\n {new_other}")
        for s in self.sources:
            if s in other.sources:
                if self.sources[s] != other.sources[s]:
                    try:
                        new_other.replace_node(other.sources[s], self.sources[s])
                    except AssertionError:
                        new_node = max(new_self.nodes) + 1
                        new_other.replace_node(new_other.sources[s], new_node)
                        new_self.replace_node(self.sources[s], new_node)
                    logger.debug(f"updated to \n{new_other}")
                keep.append(new_self.sources[s])

        # if we need new node names, they can't already be in self
        new_node = max(new_self.nodes) + 1
        to_change = [n for n in new_other.nodes if n not in keep]
        if len(to_change) > 0: logger.debug(f"changing\n {new_other}")
        for n in to_change:
            new_other.replace_node(n, new_node)
            new_node += 1
            logger.debug(f"updated to \n{new_other}")

        # combine everything
        new_nodes = new_self.nodes.union(new_other.nodes)
        logger.debug(f"merging {new_self.edges} {new_other.edges}")
        # combine lists of edges
        new_edges = merge_dicts(new_self.edges, new_other.edges)
        try:
            new_sources = merge_dicts(new_self.sources, new_other.sources)
        except SetError:
            raise GraphError(f"conflicting sources {new_self.sources} and {new_other.sources}")
        try:
            new_node_labels = merge_dicts(new_self.node_labels,
                                          new_other.node_labels)  # self.node_labels | other.node_labels
        except SetError:
            bad_labelling = ""
            for n in new_self.node_labels:
                if n in new_other.node_labels and new_self.node_labels[n] != new_other.node_labels[n]:
                    bad_labelling += f"node {n}: self: {new_self.node_labels[n]}, other: {new_other.node_labels[n]}\n"
            raise GraphError(f"conflicting node labels: {bad_labelling}")
        return type(self)(nodes=new_nodes, edges=new_edges, node_labels=new_node_labels, sources=new_sources,
                      root=new_self.root)

    def forget(self, source):
        """
        remove source (keep node)
        @param source:
        """
        if source in self.sources:
            self.sources.pop(source)
        else:
            logger.warning(f"No {source}-source to forget")

    def rename(self, old_source, new_source):
        """
        change source
        @param old_source:
        @param new_source:
        """
        if new_source in self.sources:
            raise GraphError(f"Can't rename {old_source} to {new_source}: {new_source} already exists")
        if old_source in self.sources:
            node = self.sources.pop(old_source)
            self.sources[new_source] = node

    def add_source(self, node, source):
        """
        add source to node
        @param source:
        """
        if source in self.sources:
            raise GraphError(f"{source} is already present in the graph")
        self.sources[source] = node

    def to_graphviz(self):
        ret = "digraph g {\n"
        for node in self.nodes:
            ret += f"{node}"
            label = self.get_node_label(node)
            sources = self.get_source_for_node(node)
            if label or sources or self.is_root(node):

                ret += " ["
                if self.is_root(node):
                    ret += "style=bold, "
                if label or sources:
                    ret += "label=\""
                    if label:
                        ret += f"{label}"
                    if sources:
                        ret += f"<{','.join(sources)}>"
                ret += "\"]"
            ret += ";\n"

        for node in self.edges:
            for target, label in self.edges[node]:
                ret += f"{node}->{target} [label={label}];\n"

        ret += "}"
        return ret

    def spellout(self):
        """
        For MGs
        @return:
        """
        return repr(self)


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




def apply(source, head: SGraph, argument: SGraph):
    """
    implements the AM Apply operation, with no typing
    @param source: str: the source to put the root of argument into
    @param head: the head and functor of the operation (the one with the relevant source)
    @param argument: the argument (the one to be added to head)
    @return: SGraph
    """
    logger.debug(f"App{source}\n{head}\n {argument}\n")
    assert isinstance(head, SGraph), f"head must be an SGraph but is a {type(head)}"
    assert isinstance(argument, SGraph), f"argument must be an SGraph but is a {type(head)}"
    assert source in head.sources, f"App_{source} is not possible without {source} in the head. Head: {head}"
    # don't want to copy the whole thing just to add the source, so we add it in place and then remove it
    argument.add_source(argument.root, source)
    logger.debug(f"Updated argument: {argument}")
    new_graph = head + argument
    new_graph.forget(source)
    argument.forget(source)
    return new_graph


def modify(source, head: SGraph, modifier: SGraph):
    """
     implements the AM Apply operation, with no typing
     @param source: str: the source to put the root of argument into
     @param head: the graph to have a modifier added. Will keep this one's root
     @param modifier: the modifier to be added (functor; i.e. the one with the source)
     @return: SGraph
     """

    # use a special temporary source name
    tmp = "TMP"
    assert isinstance(head, SGraph), f"head must be an SGraph but is a {type(head)}"
    assert isinstance(modifier, SGraph), f"modifier must be an SGraph but is a {type(head)}"
    assert tmp not in head.sources and tmp not in modifier.sources
    assert source in modifier.sources, f"Mod_{source} is not possible without {source} in the modifier"


    # We want to merge the root of the head and the given source of the modifier
    # don't want to copy the whole thing just to add the source, so we add it in place and then remove it
    head.add_source(head.root, tmp)
    modifier.rename(source, tmp)

    new_graph = head + modifier
    new_graph.forget(tmp)
    head.forget(source)
    modifier.rename(tmp, source)
    return new_graph


if __name__ == "__main__":
    like = SGraph(
        {0, 1, 2},
        {
            1: [(0, "ARG0"), (2, "ARG1")]
        },
        {1: "like"},
        {"S": 0, "O": 2},
        1
    )

    tried = SGraph(
        {0, 1, 2},
        {
            1: [(0, "ARG0"), (2, "ARG1")]
        },
        {1: "try"},
        {"S": 0, "O": 2},
        1
    )

    mary = SGraph(
        {0, 1},
        edges={0: [(1, "name")]},
        node_labels={0: "person", 1: "Mary"},
        root=0
    )

    cat = SGraph(
        {0},
        node_labels={0: "cat"},
        root=0
    )

    whistling = SGraph(
        {0, 1, 2},
        {
            1: [(2, "ARG0")],
            0: [(1, "mod")]
        },
        {1: "whistle"},
        {"S": 2, "M": 0},
        1
    )

    # k = g + h
    # print(k)

    m = SGraph(
        {0, 1, 2},
        {1: [(2, "with"), (0, "mod")]},
        {0: "cute", 2: "glee"},
        sources={"M": 1},
        root=0
    )

    # p = k + m
    # print(p)
    # print(h)

    vocabulary = [mary, cat, tried, like]
    for g in vocabulary:
        print(g)

    print("AppO(like, cat)")
    like_cat = apply("O", like, cat)
    print(like_cat)

    print("AppO(tried, like_cat)")
    tried_like_cat = apply("O", tried, like_cat)
    print(tried_like_cat)

    print("ModM(tried_like_cat, whistling)")
    try_whistling = modify("M", tried_like_cat, whistling)
    print(try_whistling)

    print("AppS(tried_like_cat, mary)")
    full = apply("S", try_whistling, mary)
    print(full)

    print("ModM(full, m)")
    modified = modify("M", full, m)
    print(modified)

    print(modified.to_graphviz())
