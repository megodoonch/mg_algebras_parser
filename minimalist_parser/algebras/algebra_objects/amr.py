from .graphs import SGraph
import logging


def make_amr_predicate(label: str, arg_numbers: list[int]) -> SGraph:
    """
    Make a default sort of predicate in which:
        - the root is labelled with the given label
        - the edges, if any, are labeled with ARGi for each i in arg_numbers
        - the node at the end of the lowest ARGi is an "S"-source
        - the node at the end of the second-lowest ARGi is an "O"-source
            - e.g. if arg_numbers = [0,2], edges will be ARG0 and ARG2 but sources will be S, O.
        - any additional arguments are f"O{j}"-sources for j in [2, 3, ...]
            - e.g. if there are four arg_numbers, sources should be S, O, O2, and O3
    @param label: root node label.
    @param arg_numbers: list of integers for edge labels ARG{i}.
    @return: SGraph
    """
    edge_labels = sorted([f"ARG{n}" for n in arg_numbers])
    nodes = range(len(arg_numbers))
    sources = {}
    if len(arg_numbers) > 0:
        sources = {"S": nodes[0]}
    if len(arg_numbers) > 1:
        sources["O"] = nodes[1]
        if len(arg_numbers) > 2:
            for i, node in enumerate(nodes[2:]):
                sources[f"O{i}"] = node
    root = len(arg_numbers)
    edges = {root: []}
    for l, t in zip(edge_labels, nodes):
        edges[root].append((t, l))
    nodes = set(nodes)
    nodes.add(root)
    ret = SGraph(root=root, edges=edges, sources=sources, nodes=nodes, node_labels={root: label})
    logging.debug(f"Built predicate {ret}")
    return ret


def make_amr_named_entity(entity_type: str, name_components: list[str], wiki: str = None):
    """
    Create AMR graph constant for named entity.
    :param entity_type: str: "person", "city", etc.
    :param name_components: list of parts of the name, e.g. ["Barack", "Obama"].
    :param wiki: str: Wikipedia entry, e.g. "Barack_Obama". Optional.
    :return: SGraph.
    """
    op_labels = sorted(f"op_{i+1}" for i in range(len(name_components)))
    node_labels = {0: entity_type,
                   1: "name",
                   }

    edges = {0:  [(1, "name")], 1: []}
    for i, label in enumerate(name_components):
        node_labels[i+2] = label
        edges[1].append((i+2, op_labels[i]))

    if wiki is not None:
        node_labels[2 + len(op_labels)] = wiki
        edges[0] = (2 + len(op_labels), "wiki")

    nodes = set(node_labels)

    return SGraph(root=0, edges=edges, sources={}, nodes=nodes, node_labels=node_labels)

