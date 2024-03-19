"""
For getting from a minimalist algebra term to a list of parser actions
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Dict

from minimalist_parser.algebras.algebra import Algebra
from minimalist_parser.algebras.hm_algebra import HMAlgebra
from minimalist_parser.algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from minimalist_parser.algebras.hm_triple_algebra import HMTripleAlgebra
from minimalist_parser.convert_mgbank.mgbank2algebra import add_addresses, add_intervals
from minimalist_parser.minimalism.minimalist_algebra import Expression
from minimalist_parser.minimalism.minimalist_algebra_synchronous import SynchronousTerm, MinimalistAlgebraSynchronous, \
    MinimalistFunctionSynchronous
from minimalist_parser.minimalism.movers import DSMCMovers
from minimalist_parser.minimalism.prepare_packages.addressed_triple_prepare_package import \
    HMAddressedTriplesPreparePackages
from minimalist_parser.minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare
from minimalist_parser.shift_reduce.special_vocabulary import silent_vocabulary

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

OP_CODE = 0
WORD_CODE = 1
SILENT_CODE = 2
SHIFT = "Shift"


def get_ops_and_parse_items(term: SynchronousTerm, op_state_copy_list):
    """
    Given a synchronous term with an interval pair algebra interpretation,
    get a list of triple: (minimalist op, children as parse items, list of how many copies we need of each child)
    @param term: SynchronousTerm.
    @param op_state_copy_list: list being updated.
    @return: list of pairs of (SynchronousMinimalistFunction,
                                    list of children)
    """
    if not term.is_leaf():
        op = term.parent
        op_state_copy_list.append((op, term.children))
        for child in term.children:
            get_ops_and_parse_items(child, op_state_copy_list)
    return term, op_state_copy_list


def find_child_in_stack(child, stack):
    if child.is_lexical() and len(child.inner_term.parent.function) == 0:
        return child.inner_term.parent.name
    try:
        return stack.index(child)
    except ValueError:
        found = [i for i, stack_item in enumerate(stack) if child.equal_modulo_conj(stack_item)]
        if len(found) > 0:
            return found[0]
        else:
            return

#
# def mark_duplicates(l, new_list):
#     """
#     TODO handle copies of single words.`
#     @param l:
#     @param new_list:
#     @return:
#     """
#     if len(l) == 1:
#         new_list.append((l[0], 0))
#         return new_list
#     else:
#         item = l[0]
#         c = l[1:].count(item)
#         logger.debug(f"found {c} copies of {item}")
#         new_list.append((item, c))
#         return mark_duplicates(l[1:], new_list)
#
#
# def term2actions(term: SynchronousTerm,
#                  hm_algebra: HMAlgebra,
#                  interval_algebra: HMIntervalPairsAlgebra,
#                  address_algebra: HMAlgebra):
#     # use the addresses interpretation to see where in the tree each word came from.
#     address_output = term.spellout(address_algebra)
#     _, acts = get_ops_and_parse_items(term, [])
#     acts.reverse()  # start at the bottom of the tree
#     # if any words have more than one address, they will be paired with the number of additional addresses
#     duplicates_list = mark_duplicates(acts, [])
#
#     # initialise stack with unary intervals.
#     stack = [Expression(interval_algebra.make_leaf(i), DSMCMovers()) for i in range(len(address_output))]
#     logger.debug(stack)
#     actions = []
#     for (op, children), copies in duplicates_list:
#         logger.debug(f"handling {op} {children} (with {copies} copies)")
#         logger.debug(f"Stack: {stack}")
#         interpreted_children = []  # Expressions with Interval Algebra inner terms
#         for child in children:
#             kid = child.interp(interval_algebra)
#             if kid.inner_term.parent.name == "None":
#                 # this is silent. Use the string name.
#                 kid.inner_term.parent.name = child.interp(hm_algebra).inner_term.parent.name
#             interpreted_children.append(kid)
#
#         # If they are already on the stack, get the index, otherwise, this is silent, just get the name.
#         child_stack_indices = [find_child_in_stack(child, stack) for child in interpreted_children]
#
#         logger.debug(f"found {child_stack_indices}")
#         actions.append((op, child_stack_indices))
#
#         # apply the operation and get a new Expression
#         result = op.single_function(interval_algebra).function([child.interp(interval_algebra) for child in children])
#         logger.debug(f"result: {result}")
#         if copies > 0:
#             for i, child in enumerate(children):
#                 if child.is_leaf():
#                     logger.debug(f"copy {child}")
#                     # Append the appropriate number of copies to the end of the stack
#                     stack += [stack[child_stack_indices[i]]] * copies
#         if not isinstance(child_stack_indices[0], int):
#             # if this is silent, we'll put this in the stack location of the selectee.
#             ind = child_stack_indices[1]
#         else:
#             # otherwise, put it in the head's place in the stack.
#             ind = child_stack_indices[0]
#             if len(children) > 1:
#                 # if this was Merge, replace the selectee with None
#                 stack[child_stack_indices[1]] = None
#         stack[ind] = result
#     return stack, actions
#
#
# def rename_leaves_to_indices(term: SynchronousTerm, interval_algebra):
#     if term.is_leaf():
#         if not term.parent.name.startswith("["):
#             term.parent.name = term.parent.inner_ops[interval_algebra].op_name
#     else:
#         for child in term.children:
#             rename_leaves_to_indices(child, interval_algebra)


def write_actions_to_file(path_to_file, sentence, actions, node_list, op_or_word_or_silent, adjacency_list):
    """
    Appends one sentence and its action list to the file.
    printed format:
            sentence
            \t actions as operations and stack indices
            \t node_list: list of node indices in the tree
            \t op_or_word_or_silent: list of 0,1,2 corresponding to node_list,
                                        telling us whether each is an operation, a word, or silent
            \t adjacency_list: parent child parent child... where the parent and child are list indices in the node list
            \n
    Essentially, the input to this function, in the order given, in tab-separated flat lists,
     since we can tell by where we are in the list what each item means.
    @param op_or_word_or_silent:
    @param node_list:
    @param sentence: str: input sentence.
    @param actions: list of minimalist operations, stack indices, and silent heads.
                        e.g. Merge 1 3 Move 4 means Merge items 1 and 3 on the stack then apply move to item 4.
    @param adjacency_list: list of (int, int) pairs, (parent, child) describing the whole tree with the node indices.
                            Note this is printed as a flat list as well.
    @param path_to_file: str: path to file in which to append the entry.
    """
    with open(path_to_file, "a") as f:
        s = f"{sentence}"
        for param in [actions, node_list, op_or_word_or_silent, adjacency_list]:
            s += f"\t{' '.join(str(x) for x in param)}"
        s += "\n"
        f.write(s)


def term2file(term: SynchronousTerm,
              string_alg: HMAlgebra,
              interval_algebra: HMIntervalPairsAlgebra,
              silent_vocabulary_list: list[str],
              path_to_file):
    """
    Given the complete SynchronousTerm, generates a list parser actions that result in the term, and writes them to file
    See write_actions_to_file for more on output structure.
    @param silent_vocabulary_list: list of strings: silent heads like "[past]"
    @param term: SynchronousTerm with an algebra over strings and an algebra over PairItems.
                    PairItems are the structures in parse items, e.g. (1,2),(4,6):{A: (0,1)
    @param string_alg: HMAlgebra with a spellout method outputting strings.
    @param interval_algebra: HMIntervalPairsAlgebra.
    @param path_to_file: str: path to file to append to.
    """
    sent = term.spellout(string_alg)
    sent_interval = term.spellout(interval_algebra)
    steps = term2stack_info(term, len(sent_interval), silent_vocabulary_list, interval_algebra)

    # We also want these by tree index, because this is a multidominant graph,
    # with reentrancies at ATB leaves and repeated silent heads.
    # We'll use the by-address version for a quick and dirty way to find children,
    # and the by-tree-index version to get the node list we actually want.
    steps_by_tree_index = stack_info_by_tree_index(steps)
    initial_stack = [Expression(interval_algebra.make_leaf(i), DSMCMovers()) for i in range(len(sent_interval))]

    actions = stack_info2actions(initial_stack, steps, silent_vocabulary)
    node_list = sorted(steps_by_tree_index.keys())
    op_or_word_or_silent = []
    node_label_list = []
    for node in node_list:
        info = steps_by_tree_index[node]
        category = info.is_op_or_index_or_name
        op_or_word_or_silent.append(category)
        if category in {OP_CODE, SILENT_CODE}:
            node_label_list.append(info.op.name)
        elif category == WORD_CODE:
            node_label_list.append(info.node_number)  # this is the string index
        else:
            raise ValueError('Unexpected category {}'.format(category))

    adjacency_list = []
    for address in steps:
        i = node_list.index(steps[address].node_number)
        try:
            # first child, if any
            child_tree_index = steps[address + "0"].node_number
            child_node_list_index = node_list.index(child_tree_index)
            adjacency_list += [i, child_node_list_index]
            try:
                # second child, if any
                child_tree_index = steps[address + "1"].node_number
                child_node_list_index = node_list.index(child_tree_index)
                adjacency_list += [i, child_node_list_index]
            except KeyError:
                continue
        except KeyError:
            continue

    write_actions_to_file(path_to_file, sent, actions, node_label_list, op_or_word_or_silent, adjacency_list)


def add_conj_features(sentence: str, stack: list[Expression], conjunction_list: list[str]):
    """
    Adds +conj to any stack items that could possibly be conjunctions.
    Use this on both the initial stack and the buffer when we initialise the parser.
    @param sentence: str
    @param stack: list of lexical Expressions over PairItems
    @param conjunction_list: list of strings: words that can be conjunctions, e.g. ["and", "or", "but"]
    """
    for i, word in enumerate(sentence.split()):
        if word in conjunction_list:
            stack[i].mg_type.conj = True


def add_interval_algebra(term: SynchronousTerm,
                         minimalist_algebra: MinimalistAlgebraSynchronous,
                         hm_string_algebra: HMAlgebra,
                         address_algebra: HMTripleAlgebra,
                         address_prepare: HMAddressedTriplesPreparePackages,
                         interval_algebra: HMIntervalPairsAlgebra,
                         interval_prepare: IntervalPairPrepare,
                         needs_algebras=True):
    """
    Add interval interpretation to a SynchronousTerm.
    First adds addresses, then uses them to track the right locations for the intervals.
    We have the algebras in the parameters because we don't want to make new instances of them for each term.
    @param term: the SynchronousTerm to update.
    @param minimalist_algebra: the synchronous minimalist algebra being used.
    @param hm_string_algebra: an HMAlgebra already in the term. We will use its operations to add the addresses and intervals,
                            because they are also HM algebras.
    @param address_algebra: HMTripleAlgebra over lists of tree addresses.
    @param address_prepare: HMAddressedTriplesPreparePackages
    @param interval_algebra: HMIntervalPairsAlgebra
    @param interval_prepare: IntervalPairPrepare
    @param needs_algebras: bool: If true (default), add the Interval and Addresses algebras to the minimalist algebra.
                                    If False, they should already be there.
    @return:
    """

    if needs_algebras:
        minimalist_algebra.inner_algebras[interval_algebra] = interval_prepare
        minimalist_algebra.inner_algebras[address_algebra] = address_prepare

    add_addresses(term, hm_string_algebra, address_algebra)
    address_output = term.spellout(address_algebra)
    add_intervals(term, address_output, interval_algebra, address_algebra)


@dataclass
class StackInfo:
    """
    For convenience: a way to gather together info extracted from the tree for the shift-reduce action list.
    """
    op: MinimalistFunctionSynchronous or int or str
    node_number: int
    expression: Expression
    is_op_or_index_or_name: int
    address: str = None
    childrens_addresses: Optional[list[str]] = None
    childrens_expressions: Optional[list[Expression]] = None


def term2stack_info(term: SynchronousTerm, len_sentence: int, silents, interval_algebra: HMIntervalPairsAlgebra):
    """
    Extracts StackInfo, which is information needed by Tree LSTM stack encoder, and associates with address in tree:
        minimalist op
        create unique index for node. modulo ATB
        evaluate the term
        childrens' addresses (this is redundant with len kids and address)
        evaluated children
    @param term: SynchronousTerm
    @param len_sentence: int: so we don't have to evaluate it again
    @param silents: list of silent heads to choose from for the whole grammar.
    @param interval_algebra: HMIntervalPairsAlgebra.
    @return: dict from tree address (str) to StackInfo.
    """
    len_silents = len(silents)
    stack_info = {}  # extract_stack_info_recursively fills this up

    # leaf node numbers:
    # lexical node numbers: sentence indices
    # silent heads numbers: their indices in the list of silent heads, offset by the sentence length for uniqueness

    # the first internal node will be numbered high enough that it can't interfere with leaf node numbers
    # we'll add the most recent number to the end of the list.
    # (Can't change an immutable variable from within the other function)
    node_numbers = [len_silents + len_sentence]

    def extract_stack_info_recursively(subterm: SynchronousTerm, address: str):
        """
        Run through the tree and update the stack info and current node number.
        @param subterm: the subterm we're working on now.
        @param address: str: address of the subterm (string of 0's and 1's)
        """
        if subterm.is_leaf():
            evaluated = subterm.interp(interval_algebra)
            interval = evaluated.spellout()
            # print(interval, type(interval))
            start = interval.start
            if start is not None:
                # the node name is the index in the sentence.
                # note this gives ATB copies the same name, at the lexical level only.
                leaf_node_number = start
                category = WORD_CODE
            else:
                # use the index in the silent head list, offset by the length of the sentence
                leaf_node_number = silents.index(subterm.parent.name) + len_sentence
                category = SILENT_CODE
            stack_info[address] = StackInfo(subterm.parent,
                                            leaf_node_number,
                                            evaluated,
                                            category,
                                            address,
                                            )
        else:
            # get everything for the parent
            op = subterm.parent
            node_number = node_numbers[-1]
            expression = subterm.interp(interval_algebra)
            childrens_expressions = [child.interp(interval_algebra) for child in subterm.children]
            childrens_addresses = [address + str(i) for i, _ in enumerate(subterm.children)]

            stack_info[address] = StackInfo(
                op, node_number, expression, OP_CODE, address, childrens_addresses, childrens_expressions
            )

            # hack to be able to change unmodifiable variable
            node_numbers.append(node_numbers[-1] + 1)

            # recurse in preorder order through tree by doing left child first
            extract_stack_info_recursively(subterm.children[0], address + str(0))

            if len(subterm.children) == 2:
                extract_stack_info_recursively(subterm.children[1], address + str(1))

    extract_stack_info_recursively(term, "")
    return stack_info


def stack_info2actions(stack, stack_info: dict[str: StackInfo], silents) -> list[Any]:
    """
    Given the stack info already extracted from the term,
    create the actions, their corresponding tree indices, and the adjacency list for the tree LSTM.
    @param stack: initialised stack: list of Expressions with inner terms of an HMIntervalPairsAlgebra and DSCMMovers.
    @param stack_info: dict address: StackInfo. Everything we need extracted from the term,
                                                    organised conveniently for parsing.
    @param silents: list[str]. The same list as the parser will get.
    @return: actions, tree_indices, adjacency_list, operations, arguments:
        actions: list of MinimalistFunctionSynchronous, indices (int), silent head names (str), and Shift (str)
                    Because the arities of the functions are fixed, we can use this polish notation for simplicity.

        tree_indices: corresponding node numbers in the term for each operation or leaf.
                        -1 for Shift, word index for arg of Shift.
        adjacency_list: list of (parent node number, child node number)
        operations: list of node numbers for operations (internal nodes) in stack order.
        arguments: list of node numbers for arguments in stack order.
    """
    # We need an operation name for copying a word from the sentence.
    shift_tree_index = -1  # the MG operations have node indices for where they're used in the term, so use -1 here.


    # we'll return these three lists.
    # list of MinimalistFunctionSynchronous, indices (int), silent head names (str), and Shift (str)
    # Because the arities of the functions are fixed, we can use this polish notation for simplicity.
    actions = []

    # corresponding node numbers in the term for each operation or leaf. -1 for Shift, word index for arg of Shift.
    # tree_indices = []
    # list of (parent node number, child node number)
    # adjacency_list = []
    # list of operation node numbers in stack order
    # operations = []
    # list of argument node numbers in stack order
    # arguments = []

    logger.debug(f"Initial Stack: {stack}")

    def take_step(current_stack_info: StackInfo):
        """
        Take a step in the parser and update the returns values and the stack.
        @param current_stack_info: StackInfo for the root of the current subtree.
        """
        if current_stack_info.childrens_addresses is None:
            # we don't need to do anything with leaves
            return

        # internal nodes
        child_indices = []  # tree index, stack index list
        indices_to_remove = []  # we'll remove these when we update the stack. Doesn't include silents.

        # Go through the children, finding their indices on the current stack
        # special treatment for silent heads and words that need to be copied for ATB movement
        for child_address, child in zip(current_stack_info.childrens_addresses,
                                        current_stack_info.childrens_expressions):
            tree_index = stack_info[child_address].node_number
            ind_or_silent = find_child_in_stack(child, stack)  # stack index or silent head name
            child_in_stack = isinstance(ind_or_silent, int)
            if child_in_stack:
                # normal case
                stack_silent_or_buffer_index = ind_or_silent
                indices_to_remove.append(stack_silent_or_buffer_index)
            elif ind_or_silent in silents:
                logger.debug(f"getting silent head {ind_or_silent}")
                stack_silent_or_buffer_index = ind_or_silent  # silents.index(ind_or_silent) + len_sentence
            elif child.is_lexical:
                logger.debug(f"{child} not on stack or silents. Shifting {tree_index} from buffer.")
                # "copy" from the sentence (but actually, we already have it, so just put it on the stack
                stack.append(child)
                # record this as a Shift operation
                actions.append(SHIFT)
                # tree_indices.append(shift_tree_index)
                actions.append(tree_index)  # sentence index is the same as sentence index, and we already calculated it
                # tree_indices.append(tree_index)

                # now use it
                stack_silent_or_buffer_index = len(stack) - 1  # stack index of newly added item
                indices_to_remove.append(stack_silent_or_buffer_index)
            else:
                raise ValueError(f"{child} is not on the stack, buffer, or silent list")
            child_indices.append((tree_index, stack_silent_or_buffer_index))

        # update actions
        # parent
        actions.append(current_stack_info.op)
        # tree_indices.append(current_stack_info.node_number)
        # operations.append(current_stack_info.node_number)
        # children
        for (tree_index, child_index) in child_indices:
            actions.append(child_index)
            # tree_indices.append(tree_index)
            # adjacency_list.append((current_stack_info.node_number, tree_index))
            # arguments.append(tree_index)

        # update stack
        # remove children
        # to pop multiple things, sort the indices to remove from highest to lowest
        for i in sorted(indices_to_remove, reverse=True):
            stack.pop(i)
        # append new item to the end
        stack.append(current_stack_info.expression)

        logger.debug(f"{current_stack_info.op}({[stack_index for _, stack_index in child_indices]})")
        logger.debug(f"stack: {stack}")

    addresses_in_order = reversed(stack_info.keys())
    for address in addresses_in_order:
        take_step(stack_info[address])

    return actions   # tree_indices, adjacency_list, operations, arguments


def stack_info_by_tree_index(stack_info: dict[str: StackInfo]):
    stack_info_by_tree_index_dict = {}
    for address in stack_info:
        stack_info_by_tree_index_dict[stack_info[address].node_number] = stack_info[address]
        # just to remember that this might be more than one address, and we don't need them anyway.
        stack_info_by_tree_index_dict[stack_info[address].node_number].address = None
    return stack_info_by_tree_index_dict


def list_of_unaddressed_terms2file(terms: list[SynchronousTerm],
                                   output_file: str,
                                   minimalist_algebra: MinimalistAlgebraSynchronous,
                                   string_algebra: HMAlgebra,
                                   interval_algebra: HMIntervalPairsAlgebra,
                                   address_algebra: HMTripleAlgebra,
                                   silent_vocabulary_list: list[str],
                                   write_header: bool = False):
    """
    Wrapper function for list of terms that don't yet have an address interpretation or an interval interpretation.
    Adds the interpretaions, makes the actions, and appends everything the DatasetReader needs.
    Assumes minimalist_algebra already has the new algebras added.
    @param write_header: If true, write a header to the file so we know what it means. Default False.
    @param terms: list of SynchronousTerms
    @param output_file: where to write the data to.
    @param minimalist_algebra: MinimalistAlgebraSynchronous the terms are over.
    @param string_algebra: a string algebra that the minimalist algebra already has.
    @param interval_algebra: the interval algebra in use.
    @param address_algebra: the HMAlgebra over address triples in use.
    @param silent_vocabulary_list: list of strings: the silent vocabulary for this lexicon.
    """
    if write_header:
        with open(output_file, "w") as conn:
            conn.write("Sentence\tActions\tNode List\tOp or Word or Silent\tAdjacency List\n")
    address_prepare = minimalist_algebra.inner_algebras[address_algebra]
    interval_prepare = minimalist_algebra.inner_algebras[interval_algebra]
    for term in terms:
        add_interval_algebra(term, minimalist_algebra, string_algebra, address_algebra, address_prepare,
                             interval_algebra, interval_prepare, False)
        term2file(term, string_algebra, example_interval_algebra, silent_vocabulary_list, output_file)


if __name__ == "__main__":
    from ..examples import t3, tree_atb, tree_atb_hm, triple_alg, mg

    toy_data_path = "data/processed/shift_reduce/toy/neural_input.txt"

    # make required algebras and add to MG
    example_interval_algebra = HMIntervalPairsAlgebra()
    example_interval_prepare = IntervalPairPrepare()
    example_interval_algebra.add_constant_maker()

    example_address_algebra = HMTripleAlgebra("addresses algebra", component_type=list, addresses=True)
    example_address_prepare = HMAddressedTriplesPreparePackages()

    mg.inner_algebras[example_address_algebra] = example_address_prepare
    mg.inner_algebras[example_interval_algebra] = example_interval_prepare

    # make the actions and write  them to file
    list_of_unaddressed_terms2file([t3, tree_atb, tree_atb_hm], toy_data_path, mg, triple_alg,
                                   example_interval_algebra, example_address_algebra, silent_vocabulary, False)

    # for t in [t3, tree_atb, tree_atb_hm]:
    #     add_interval_algebra(t, mg, triple_alg, example_address_algebra, example_address_prepare, example_interval_algebra, example_interval_prepare, False)
    #     term2file(t, triple_alg, example_interval_algebra, silent_vocabulary, toy_data_path)
        # sent = t.spellout(triple_alg)
        # sent_interval = t.spellout(example_interval_algebra)
        # steps = term2stack_info(t, len(sent_interval), silent_vocabulary, example_interval_algebra)
        # print(f"\n{sent}")
        # # for step in steps:
        # #     print(f"\t{step}: {steps[step]}")
        # initial_stack = [Expression(example_interval_algebra.make_leaf(i), DSMCMovers()) for i in range(len(sent_interval))]
        # acts, inds, edges, ops, args = stack_info2actions(initial_stack, steps, silent_vocabulary)
        # print(f"\n{acts}")
        # print(f"{inds}")
        # print(edges)
        # print(ops)
        # print(args)
        # # tree_index2tree_address = {steps[addr].node_number: addr for addr in steps}
        # # print(tree_index2tree_address)
        # write_actions_to_file(sent, acts, inds, edges, ops, args, "data/processed/shift_reduce/toy/lstm_input.txt")


        # term2file(t, triple_alg, example_interval_algebra, example_address_algebra, toy_data_path)
