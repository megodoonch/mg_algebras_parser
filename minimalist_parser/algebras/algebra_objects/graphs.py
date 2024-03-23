import logging
import sys
from copy import deepcopy
from typing import Set, Iterable
import penman

from ..algebra import AlgebraError
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format='%(levelname)s (%(name)s) - %(message)s')


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
        assert nodes is None or isinstance(nodes, Set), f"Nodes must be of type Set but is {type(nodes)}"
        assert edges is None or isinstance(edges, dict), f"Edges must be of type dict but is {type(edges)}"
        assert node_labels is None or isinstance(node_labels,
                                                 dict), f"node_labels must be of type dict but is {type(node_labels)}"
        assert sources is None or isinstance(sources, dict), f"sources must be of type dict but is {type(sources)}"
        assert root is None or isinstance(nodes, Iterable) and root in nodes, f"root must be in nodes"

        self.nodes = set() if nodes is None else nodes
        self.edges = {} if edges is None else edges
        self.node_labels = {} if node_labels is None else node_labels
        self.sources = {} if sources is None else sources
        self.root = root

        assert set(self.edges.keys()).issubset(self.nodes), f"Edges must be subset of nodes x nodes"
        assert set(self.sources.values()).issubset(self.nodes), f"Sources must be subset of nodes"
        assert set(self.node_labels.keys()).issubset(self.nodes), \
            f"Node labeling refers to non-existent nodes: {set(self.node_labels.keys())} vs {self.nodes}"

    def get_source_for_node(self, n):
        sources = []
        for source in self.sources:
            if n == self.sources[source]:
                sources.append(source)
        return sources

    def __str__(self):
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
        Adds two graphs together, keeping the root at the root of self, and merging shared sources.
        :param other: SGraph
        :return: SGraph
        """
        assert isinstance(other, type(self))
        # copy both graphs so we don't mess with the originals
        new_self = deepcopy(self)
        new_other = deepcopy(other)

        # rename all nodes in other
        new_node = max(self.nodes.union(other.nodes)) + 1  # avoid all possible conflicts of node names
        for node in other.nodes:
            new_other.replace_node(node, new_node)
            new_node += 1
        # if self and other share any sources, make them the same in other as they are in self.
        for source in self.sources:
            if source in other.sources:
                new_other.replace_node(new_other.sources[source], self.sources[source])

        # update copy of self to include everything in other
        new_self.nodes = self.nodes.union(new_other.nodes)
        new_self.edges.update(new_other.edges)
        new_self.sources.update(new_other.sources)
        new_self.node_labels.update(new_other.node_labels)
        return new_self

    def forget(self, source: str):
        """
        remove source (keep node)
        @param source: str
        """
        if source in self.sources:
            self.sources.pop(source)
        else:
            logger.warning(f"No {source}-source to forget")

    def rename(self, old_source: str, new_source: str):
        """
        change source
        @param old_source: str: the source to change
        @param new_source: str: the source to change to
        """
        if new_source in self.sources:
            raise GraphError(f"Can't rename {old_source} to {new_source}: {new_source} already exists")
        if old_source in self.sources:
            node = self.sources.pop(old_source)
            self.sources[new_source] = node

    def add_source(self, node, source):
        """
        add source to node.
        @param source: str: the source to assign to the node.
        @param node: the node to be given the source.
        """
        if source in self.sources:
            raise GraphError(f"{source} is already present in the graph")
        self.sources[source] = node

    def to_graphviz(self):
        """
        Make a graphviz (dot) representation of the graph.
        :return: str
        """
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

    def to_penman(self):
        triples = []
        for source, edges in self.edges.items():
            for target, label in edges:
                triples.append((source, label, target))
        for node, label in self.node_labels.items():
            triples.append((node, ":instance", label))
        for source, node in self.sources.items():
            triples.append((node, ":instance", f"<{source}>"))
        g = penman.graph.Graph(triples, top=self.root)
        return g


    def add_op_source(self):
        """
        Add an opi edge from the root. Its target is an OP source,
            and the i is the next up from whatever is already there.
        @return: None, modify in place
        """
        # see what opis we already have
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


