"""
Author: Jonas Groschwitz
UPdated by Meaghan FOwlie
"""

from typing import Dict, Any, List

import torch
import numpy

from allennlp.data import Vocabulary, Token
from allennlp.data.fields import TextField
from allennlp.data.fields.field import Field
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from minimalist_parser.convert_mgbank.term2actions import WORD_CODE, OP_CODE, SILENT_CODE


# TODO update to match my needs


class TreeField(Field):
    """
    Attributes:
        node_list: A list of operations, word_ids, and silent heads. Not really needed in this form; just for logging.
        operation_or_word_or_silent: A list of entries that each are from 0, 1, 2.
                                        Same length and order as node_features.
                                        0 means the node is labeled with an operation, 1 with word, 2 with silent.
        operations: TextField: The list of operations (in text), as a subsequence of the Node List.
                                    (I.e. the node labels of the nodes in Node List that are operations, in order)
        word_ids: TensorField: For the nodes in Node List that correspond to words, in order,
                                    the index of the word in the sentence that this node refers to.
        silent: TextField: The list of silent heads (in text), as a subsequence of the Node List.
                                    (I.e. the node labels of the nodes in Node List that are silent heads, in order)
        adjacency_list: List of lists of length 2, containing ints. Each pair is a [parent, child] index in node_list
        node_order: List of ints, corresponding to node_list. 0 means this node can be parsed right away,
            1 means we need to wait for the 0's to be done, etc.
        edge_order: List of ints, corresponding to the adjacency list.
            1 means this edge can be parsed as soon as the leaves are done, 2 means we need to wait for the 1's ,etc.
    """

    def __init__(self, node_list: list[str], adjacency_list: list[str], operation_or_word_or_silent: list[str],
                 available_stack_nodes: list[list[int]]):
        """
        Creates a TreeField containing everything the TreeLSTM needs to know about the tree.
        Parameters are all string lists, as they were read in by the DataReader.
        @param node_list: List of node labels, or sentence indices for words.
        @param adjacency_list: flat list of indices in the node_list, [parent, child, parent, child...]
        @param operation_or_word_or_silent: List of 0,1,2, according as the node in the node_list
                                            is an operation, word index, or silent head.
        @param available_stack_nodes: List of lists of indices of entries in node_list.
        """
        # print("INIT")
        # print(am_sentence)
        super().__init__()  # node_tokenizer, word_from_node_token_indexer, node_token_indexer,
        # token_index_to_selection_sequence_index, selection_sequence_index_to_token_index)
        # self.am_sentence = am_sentence
        # self.node_linearizer = node_linearizer
        # self.tree = create_unbounce_style_tree(am_sentence, node_tokenizer, node_token_indexer,
        # word_from_node_token_indexer, node_linearizer)
        # print(self.tree)
        # self.flat_tree = None

        self.node_list = [int(node) if node.isdigit() else node for node in node_list]

        self.operation_or_word_or_silent = [int(x) for x in operation_or_word_or_silent]
        self.adjacency_list = []
        # un-flatten the adjacency list, currently just [parent child parent child...]
        for i in range(len(adjacency_list))[:-1:2]:
            self.adjacency_list.append([int(adjacency_list[i]), int(adjacency_list[i + 1])])

        # unpack the nodes into their respective categories
        self.word_ids = []
        self.operations = []
        self.silent = []
        for category, node in zip(self.operation_or_word_or_silent, self.node_list):
            if category == OP_CODE:
                self.operations.append(Token(text=node))
            elif category == WORD_CODE:
                self.word_ids.append(node)
            elif category == SILENT_CODE:
                self.silent.append(Token(text=node))
            else:
                raise ValueError(f"category can only be 0 (operation), 1 (word), or 2 (silent), but got {category}")

        self.node_order, self.edge_order = calculate_evaluation_orders(self.adjacency_list, len(node_list))

        # make ops and silents into TextFields
        self.operations = TextField(self.operations,
                                    token_indexers={"tokens": SingleIdTokenIndexer(namespace=OP_NAMESPACE)})
        self.silent = TextField(self.silent,
                                token_indexers={"tokens": SingleIdTokenIndexer(namespace=SILENT_NAMESPACE)})

        self.available_stack_nodes = available_stack_nodes

    def __repr__(self):
        s = "TreeField:\n"
        s += f"\toperation_or_word_or_silent: {self.operation_or_word_or_silent}\n"
        s += f"\tnode_list: {self.node_list}\n"
        s += f"\tadjacency_list: {self.adjacency_list}\n"
        s += f"\tword_ids: {self.word_ids}\n"
        s += f"\toperations: {self.operations}\n"
        s += f"\tsilent: {self.silent}\n"
        s += f"\tnode_order: {self.node_order}\n"
        s += f"\tedge_order: {self.edge_order}\n"
        s += f"\tavailable_stack_nodes: {self.available_stack_nodes}\n"
        return s

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        one of the required methods for a Field
        @param counter:
        """
        self.operations.count_vocab_items(counter)
        self.silent.count_vocab_items(counter)

    def index(self, vocab: Vocabulary):
        """
        required for Field
        Turns words into indices
        @param vocab:
        """
        self.operations.index(vocab)
        self.silent.index(vocab)
        # words are already numbers

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        Required for Field.
        We're concatenating the trees so we don't need actual padding for them.
        But this might get used for grouping things of similar size together, so we return the tree size.
        @return: dict: str to int
        """
        return {"number_of_nodes": len(self.node_list), "number_of_arguments": len(self.available_stack_nodes)}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> dict[str, torch.Tensor]:
        """
        Required for field.
        Turns tree (as represented by vocab indices) into tensors, padding if nec
        @param padding_lengths:
        @return:
        """
        ret = {}
        ret["word_ids"] = torch.tensor(self.word_ids, dtype=torch.long)

        # extract the indices of the operations, as indexed by TextField
        # and turn them into a tensor
        ret["operations"] = torch.tensor(self.operations._indexed_tokens["tokens"]["tokens"], dtype=torch.long)
        ret["silent"] = torch.tensor(self.silent._indexed_tokens["tokens"]["tokens"], dtype=torch.long)

        # these are all already numbers
        ret["operation_or_word_or_silent"] = torch.tensor(self.operation_or_word_or_silent, dtype=torch.long)
        ret["adjacency_list"] = torch.tensor(self.adjacency_list, dtype=torch.long)
        ret["node_order"] = torch.tensor(self.node_order, dtype=torch.long)
        ret["edge_order"] = torch.tensor(self.edge_order, dtype=torch.long)

        # it may be important to make this a list of tensors here, to have it moved to the correct device etc.
        # (not quite sure if allennlp does this before or after batching)
        # iirc, allennlp just iterates over lists (of lists of ...) of tensors automatically
        # we can still do the offsetting and the padding later
        # I think every entry in ret should be a tensor, or a list of tensors etc.
        ret["available_stack_nodes"] = [torch.tensor(v, dtype=torch.long) for v in self.available_stack_nodes]

        return ret

    def batch_tensors(self, tensor_list: List):  # type: ignore
        """
        Required for field.
        given a list of TreeFields as tensors (output of TreeField.as_tensor),
        batch the trees.
        @param tensor_list:
        @return:
        """
        batched_node_order = torch.cat([sentence["node_order"] for sentence in tensor_list])
        batched_edge_order = torch.cat([sentence["edge_order"] for sentence in tensor_list])
        batched_adjacency_list = []  # will offset by previous tree sizes
        batched_available_stack_nodes = []  # will offset "
        offset = 0
        avail_nodes_padding_value = -1
        for sentence in tensor_list:
            batched_adjacency_list.append(sentence["adjacency_list"] + offset)  # pointwise adding offset

            batched_available_stack_nodes_here = self.batch_and_offset_available_stack_nodes_for_sentence(
                avail_nodes_padding_value, offset, sentence)
            batched_available_stack_nodes.append(batched_available_stack_nodes_here)

            offset += sentence["operation_or_word_or_silent"].shape[0]  # number of nodes in the graph

        batched_available_stack_nodes = self.pad_and_stack_available_stack_nodes_on_batch_level(avail_nodes_padding_value,
                                                                                                batched_available_stack_nodes)
        batched_adjacency_list = torch.cat(batched_adjacency_list)
        batched_operations = torch.cat([sentence["operations"] for sentence in tensor_list])
        batched_silent = torch.cat([sentence["silent"] for sentence in tensor_list])

        # the word_ids are used eventually for look-up in a tensor of shape
        #  batch_size x sentence_length x hidden_dim.
        # So we want to make a list of tuples, where the first item in the tuple tells us the position in the batch
        #  (i.e. the sentence ID in the batch; named i below)
        # and the second item in the tuple tells us the position in the sentence (i.e. the word_id).
        word_ids_with_sentence_index = []
        for i, sentence in enumerate(tensor_list):
            word_ids_here = sentence["word_ids"]
            # make a tensor with entries all equal to i, with the same shape as word_ids_here
            sentence_id_tensor = torch.tensor([i] * word_ids_here.shape[0], dtype=torch.long)
            # stack in dimension 1 because we overall want to make a list of pairs, not a pair of lists.
            # put the sentence_id_tensor first and word_ids_here second, because in the later tensor where we
            # will do the lookup, the sentence indices come first and then the word indices
            # (i.e. the shape is batch_size x sentence_length x hidden_dim)
            word_ids_with_sentence_index.append(torch.stack([sentence_id_tensor, word_ids_here], dim=1))
        batched_word_ids_with_sentence_index = torch.cat(word_ids_with_sentence_index, dim=0)

        batched_operation_or_word_or_silent = torch.cat([b["operation_or_word_or_silent"] for b in tensor_list])

        ret = {}
        ret["word_ids"] = batched_word_ids_with_sentence_index
        ret["operations"] = batched_operations
        ret["silent"] = batched_silent
        ret["operation_or_word_or_silent"] = batched_operation_or_word_or_silent
        ret["adjacency_list"] = batched_adjacency_list
        ret["node_order"] = batched_node_order
        ret["edge_order"] = batched_edge_order
        ret["available_stack_nodes"] = batched_available_stack_nodes

        return ret


        # tree_sizes = [b.structure.is_nodes.shape[0] for b in batch]
        # batched_edge_labels = torch.cat([b.structure.edge_labels for b in batch])
        # batched_is_nodes = torch.cat([b.structure.is_nodes for b in batch])
        # batched_reverse_node_order = torch.cat([b.structure.reverse_node_order for b in batch])
        # batched_edge_order = torch.cat([b.structure.edge_order for b in batch])
        # batched_reverse_edge_order = torch.cat([b.structure.reverse_edge_order for b in batch])
        # batched_adjacency_list = []
        # batched_reverse_adjacency_list = []
        # offset = 0
        # for n, sentence in zip(tree_sizes, batch):
        #     batched_adjacency_list.append(sentence.structure.adjacency_list + offset)
        #     batched_reverse_adjacency_list.append(sentence.structure.reverse_adjacency_list + offset)
        #     offset += n
        # batched_adjacency_list = torch.cat(batched_adjacency_list)
        # batched_reverse_adjacency_list = torch.cat(batched_reverse_adjacency_list)
        # tree_structure = TreeStructure(
        #     edge_labels=batched_edge_labels,
        #     is_nodes=batched_is_nodes,
        #     node_order=batched_node_order,
        #     reverse_node_order=batched_reverse_node_order,
        #     edge_order=batched_edge_order,
        #     reverse_edge_order=batched_reverse_edge_order,
        #     adjacency_list=batched_adjacency_list,
        #     reverse_adjacency_list=batched_reverse_adjacency_list,
        #     tree_sizes=tree_sizes
        # )
        #
        # print(tensor_list)
        # return batch_tree_input(tensor_list)

    def pad_and_stack_available_stack_nodes_on_batch_level(self, avail_nodes_padding_value, batched_available_stack_nodes):
        # each sentence in the batch has a list of SR arguments
        max_num_sr_arguments = max(t.shape[0] for t in batched_available_stack_nodes)
        # we actually pad the "available stack nodes" dimension twice
        # once sentence-internally, in the for loop above (so the max within each sentence is already t.shape[1])
        # and now again batch-wide, here
        max_avail_stack_nodes = max(t.shape[1] for t in batched_available_stack_nodes)
        # Pad
        # max_avail_stack_nodes - t.shape[1] because this sentence is only t.shape[1] args, so the rest is padding
        # note that the "pad" argument (a tuple) goes through the dimensions *backwards*, and for each dimension
        # has two numbers, the first specifying padding on the left (we don't use that, hence the 0), the second padding
        # on the right.
        batched_available_stack_nodes = [torch.nn.functional.pad(t, (0, max_avail_stack_nodes - t.shape[1],
                                                                     0, max_num_sr_arguments - t.shape[0]),
                                                                 value=avail_nodes_padding_value)
                                         for t in batched_available_stack_nodes]
        batched_available_stack_nodes = torch.stack(batched_available_stack_nodes)
        return batched_available_stack_nodes

    def batch_and_offset_available_stack_nodes_for_sentence(self, avail_nodes_padding_value, offset, sentence):
        batched_available_stack_nodes_here = []
        max_choices = max(t.shape[0] for t in sentence["available_stack_nodes"])
        for t in sentence["available_stack_nodes"]:
            t_new = self.offset_and_pad_available_nodes_locally(avail_nodes_padding_value, max_choices, offset, t)
            batched_available_stack_nodes_here.append(t_new)
        batched_available_stack_nodes_here = torch.stack(batched_available_stack_nodes_here)
        return batched_available_stack_nodes_here

    def offset_and_pad_available_nodes_locally(self, avail_nodes_padding_value, max_choices, offset, t):
        t_new = t + offset
        t_new = torch.nn.functional.pad(t_new, (0, max_choices - t_new.shape[0]),
                                        value=avail_nodes_padding_value)
        return t_new

    def empty_field(self) -> "Field":
        """
        Rewuired but this should be fine
        @return:
        """
        pass

    def __len__(self):
        """
        Returns the number of nodes in the tree. (Only real nodes, not the ones that are technically nodes in this
         object, but really represent edges)
        :return:
        """
        return len(self.node_list)

    def human_readable_repr(self) -> Any:
        return repr(self)

    def convert_flat_tree_to_tensors(self, max_node_sequence_length, max_word_length):
        """

        :param max_word_length:
        :param max_node_sequence_length:
        :return:
        """
        pass
        # node_count = len(self.flat_tree['node_labels'])
        # tree_structure = TreeStructure(
        #     edge_labels=torch.tensor(self.flat_tree['edge_labels'], dtype=torch.long),
        #     is_nodes=torch.tensor(self.flat_tree['is_nodes'], dtype=torch.bool),
        #     node_order=torch.tensor(self.flat_tree['node_order'], dtype=torch.long),
        #     reverse_node_order=torch.tensor(self.flat_tree['reverse_node_order'], dtype=torch.long),
        #     edge_order=torch.tensor(self.flat_tree['edge_order'], dtype=torch.long),
        #     reverse_edge_order=torch.tensor(self.flat_tree['reverse_edge_order'], dtype=torch.long),
        #     adjacency_list=torch.tensor(self.flat_tree['adjacency_list'], dtype=torch.long),
        #     reverse_adjacency_list=torch.tensor(self.flat_tree['reverse_adjacency_list'], dtype=torch.long),
        #     tree_sizes=[node_count]
        # )
        #
        # aligned_structure = self.create_aligned_structure(max_node_sequence_length, max_word_length, node_count,
        #                                                   tree_structure)
        #
        # return aligned_structure

        # return {
        #     'edge_labels': torch.tensor(self.flat_tree['edge_labels'], dtype=torch.long),
        #     'node_labels': [text_field.as_tensor(padding_lengths_dict_nodes) for text_field
        #                     in self.flat_tree['node_labels']],
        #     'aligned_texts': [text_field.as_tensor(padding_lengths_dict_words) for text_field
        #                       in self.flat_tree['aligned_texts']],
        #     'is_nodes': torch.tensor(self.flat_tree['is_nodes'], dtype=torch.bool),
        #     'node_order': torch.tensor(self.flat_tree['node_order'], dtype=torch.long),
        #     'reverse_node_order': torch.tensor(self.flat_tree['reverse_node_order'], dtype=torch.long),
        #     'adjacency_list': torch.tensor(self.flat_tree['adjacency_list'], dtype=torch.long),
        #     'reverse_adjacency_list': torch.tensor(self.flat_tree['reverse_adjacency_list'], dtype=torch.long),
        #     'edge_order': torch.tensor(self.flat_tree['edge_order'], dtype=torch.long),
        #     'reverse_edge_order': torch.tensor(self.flat_tree['reverse_edge_order'], dtype=torch.long),
        #     'selection_sequence': selection_sequence_tensor,
        #     'next_selection_sequence': next_selection_sequence_tensor,
        #     'next_selection_masks': torch.tensor(self.next_selection_masks, dtype=torch.bool),
        #     'token_index_to_next_node_index': token_index_to_next_node_index,
        #     'node_index_to_token_index': torch.tensor(node_index_to_token_index, dtype=torch.long),
        #     'node_index_to_selection_sequence_index': torch.tensor(node_index_to_selection_sequence_index,
        #                                                            dtype=torch.long)
        # }

    def create_nl_and_aligned_text_tensors(self, padding_lengths_dict_nodes, padding_lengths_dict_words):
        node_labels_tensor = [text_field.as_tensor(padding_lengths_dict_nodes) for text_field
                              in self.flat_tree['node_labels']]
        aligned_texts_tensor = [text_field.as_tensor(padding_lengths_dict_words) for text_field
                                in self.flat_tree['aligned_texts']]
        return aligned_texts_tensor, node_labels_tensor


def inc_value_in_container_if_is_node(node, container):
    if node['is_node']:
        container[0] += 1


# def create_unbounce_style_tree(am_sentence, node_tokenizer, node_token_indexer, word_from_node_token_indexer,
#                                node_linearizer):
#     return create_unbounce_style_tree_recursive(am_sentence,
#                                                 get_one_based_root_index(am_sentence),
#                                                 node_tokenizer, node_token_indexer, word_from_node_token_indexer,
#                                                 node_linearizer)


# def create_unbounce_style_tree_recursive(am_sentence, current_id,
#                                          node_tokenizer,
#                                          node_token_indexer,
#                                          word_from_node_token_indexer,
#                                          node_linearizer):
#     """
#
#     :param word_from_node_token_indexer:
#     :param node_token_indexer:
#     :param node_tokenizer:
#     :param am_sentence:
#     :param current_id: is one-based
#     :return:
#     """
#     # TODO add type information!
#     # print(current_id)
#     children_ids = [i + 1 for i, w in enumerate(am_sentence) if w.head == current_id]
#     # print(children_ids)
#     edge_tree = {
#         'label': am_sentence.words[current_id - 1].label, 'is_node': False,
#         'aligned_text': None, 'position_in_sentence': None
#     }
#     node_label_tokens = node_tokenizer.tokenize(node_linearizer.linearize(am_sentence.words[current_id - 1].fragment))
#     # TODO use different tokenizer for the words EDIT: maybe not necessary
#     leading_space = "" if current_id == 1 else " "
#     word_tokens = node_tokenizer.tokenize(leading_space + am_sentence.words[current_id - 1].token)
#     log_node_example(node_label_tokens)
#     log_aligned_text_example(word_tokens)
#     if not (isinstance(node_tokenizer, PretrainedTransformerTokenizer)
#             or isinstance(node_tokenizer, WhitespacePreservingBartTokenizer)):
#         node_label_tokens = [Token(START_SYMBOL)] + node_label_tokens + [Token(END_SYMBOL)]
#         word_tokens = [Token(START_SYMBOL)] + word_tokens + [Token(END_SYMBOL)]
#     # noinspection PyDictCreation
#     node_tree = {
#         'label': TextField(node_label_tokens, token_indexers={NODE_LABEL_NAMESPACE: node_token_indexer}), 'is_node': True,
#         'aligned_text': TextField(word_tokens, token_indexers={
#             WORD_FROM_NODE_NAMESPACE: word_from_node_token_indexer}),
#         'position_in_sentence': current_id - 1
#     }
#     edge_tree['children'] = [node_tree]
#     node_tree['children'] = [create_unbounce_style_tree_recursive(am_sentence, i, node_tokenizer,
#                                                                   node_token_indexer, word_from_node_token_indexer,
#                                                                   node_linearizer)
#                              for i in children_ids]
#     return edge_tree
#
#
# def get_one_based_root_index(am_sentence):
#     return [i + 1 for i, w in enumerate(am_sentence.words) if w.label == "ROOT" and w.head == 0][0]
#
#
# def count_vocab_items_in_tree(tree, counter):
#     if tree['is_node']:
#         tree['label'].count_vocab_items(counter)
#         tree['aligned_text'].count_vocab_items(counter)
#     else:
#         counter[EDGE_LABEL_NAMESPACE][tree['label']] += 1
#     for child in tree['children']:
#         count_vocab_items_in_tree(child, counter)
#
#
# def index_tree(tree, vocab):
#     if tree['is_node']:
#         tree['label'].index(vocab)
#         tree['aligned_text'].index(vocab)
#     else:
#         tree['label'] = vocab.get_token_index(tree['label'], EDGE_LABEL_NAMESPACE)
#     for child in tree['children']:
#         index_tree(child, vocab)
#
#
# def get_depth(tree):
#     if len(tree['children']) > 0:
#         return max(get_depth(child) for child in tree['children']) + 1
#     else:
#         return 1
#
#
# def get_max_node_sequence_length(tree, key):
#     if len(tree['children']) > 0:
#         max_from_children = max(get_max_node_sequence_length(child, key) for child in tree['children'])
#     else:
#         max_from_children = 0
#     if tree['is_node']:
#         # print(tree[key].get_padding_lengths())
#         if f'{NODE_LABEL_NAMESPACE}___tokens' in tree[key].get_padding_lengths().keys():
#             lookup_string = f'{NODE_LABEL_NAMESPACE}___tokens'
#         elif f'{WORD_FROM_NODE_NAMESPACE}___tokens' in tree[key].get_padding_lengths().keys():
#             lookup_string = f'{WORD_FROM_NODE_NAMESPACE}___tokens'
#         elif f'{NODE_LABEL_NAMESPACE}___token_ids' in tree[key].get_padding_lengths().keys():
#             lookup_string = f'{NODE_LABEL_NAMESPACE}___token_ids'
#         elif f'{WORD_FROM_NODE_NAMESPACE}___token_ids' in tree[key].get_padding_lengths().keys():
#             lookup_string = f'{WORD_FROM_NODE_NAMESPACE}___token_ids'
#         else:
#             raise Exception("Could not find a good lookup string when getting padding lengths. Probably related to "
#                             "the token indexer.")
#         length_here = tree[key].get_padding_lengths()[lookup_string]
#         return max(length_here, max_from_children)
#     else:
#         return max_from_children
#
#
# # ---------------- much of the below is adapted or copied from https://github.com/unbounce/pytorch-tree-lstm
#
# def _label_node_index(node, n=0):
#     node['index'] = n
#     for child in node['children']:
#         n += 1
#         n = _label_node_index(child, n)
#     return n
#
#
# def _gather_node_attributes(node, key):
#     features = [node[key]]
#     for child in node['children']:
#         features.extend(_gather_node_attributes(child, key))
#     return features
#
#
# def _gather_adjacency_list(node):
#     adjacency_list = []
#     for child in node['children']:
#         adjacency_list.append([node['index'], child['index']])
#         adjacency_list.extend(_gather_adjacency_list(child))
#
#     return adjacency_list
#
#
def calculate_evaluation_orders(adjacency_list, tree_size):
    """Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.

    The TreeLSTM model requires node_order and edge_order to be passed into the model along
    with the node features and adjacency_list.  We pre-calculate these orders as a speed
    optimization.

    Note that this is for bottom-up evaluation, so we evaluate the leaves first, and only do a parent node
    if all its children have been processed.
    @return node_order, edge_order
    """

    parents = [p for p, _ in adjacency_list]
    children = [c for _, c in adjacency_list]
    adjacency_list = numpy.array(adjacency_list)

    node_ids = numpy.arange(tree_size, dtype=int)

    node_order = numpy.zeros(tree_size, dtype=int)
    unevaluated_nodes = numpy.ones(tree_size, dtype=bool)

    if len(adjacency_list) > 0:
        parent_nodes = numpy.array(adjacency_list[:, 0])
        child_nodes = numpy.array(adjacency_list[:, 1])
    else:
        parent_nodes = numpy.array([], dtype=int)
        child_nodes = numpy.array([], dtype=int)

    n = 0
    while unevaluated_nodes.any():
        if n > 1000:
            print(f"Too long loop in calculate_evaluation_orders {n}")
            print(unevaluated_nodes)
        # Find which child nodes have not been evaluated
        unevaluated_children_mask = unevaluated_nodes[child_nodes]

        # Find the parent nodes of unevaluated children
        parents_of_unevaluated_children = parent_nodes[unevaluated_children_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of parents with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, parents_of_unevaluated_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order


#
# def flatten(tree):
#     # Label each node with its walk order to match nodes to feature tensor indexes
#     # This modifies the original tree as a side effect
#     _label_node_index(tree)
#
#     labels = _gather_node_attributes(tree, 'label')
#     is_nodes = _gather_node_attributes(tree, 'is_node')
#     node_labels = [label for i, label in enumerate(labels) if is_nodes[i]]
#     adjacency_list = _gather_adjacency_list(tree)
#
#     node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(labels))
#
#     return {
#         'node_labels': node_labels,  # trickiness: there are 3 kinds of these.
#         'node_order': node_order,  # for tree lstm
#         'adjacency_list': adjacency_list,  # for tree lstm
#         'edge_order': edge_order,  # not tree lstm
#     }
#
#
# def batch_tree_input(batch: List[AlignedStructure]) -> AlignedStructure:
#     """Combines a batch of aligned tree structures into a single batched structure for use by the TreeLSTM model.
#
#     """
#     tree_structure = batch_tree_structure(batch)
#
#     aligned_structure = batch_aligned_structure(batch, tree_structure)
#
#     return aligned_structure
#
#     # return {
#     #     'edge_labels': batched_edge_labels,
#     #     'node_labels': batched_node_labels,
#     #     'aligned_texts': batched_aligned_texts,
#     #     'is_nodes': batched_is_nodes,
#     #     'node_order': batched_node_order,
#     #     'reverse_node_order': batched_reverse_node_order,
#     #     'edge_order': batched_edge_order,
#     #     'reverse_edge_order': batched_reverse_edge_order,
#     #     'adjacency_list': batched_adjacency_list,
#     #     'reverse_adjacency_list': batched_reverse_adjacency_list,
#     #     'tree_sizes': tree_sizes,
#     #     'selection_mask': selection_mask,
#     #     'selection_sequence': batched_selection_sequence,
#     #     'next_selection_sequence': batched_next_selection_sequence,
#     #     'next_selection_masks': batched_next_selection_masks,
#     #     'complete_or_inflect': complete_or_inflect,
#     #     'token_index_to_next_node_index': batched_token_index_to_next_node_index,
#     #     'node_index_to_token_index': batched_node_index_to_token_index,
#     #     'node_index_to_selection_sequence_index': batched_node_index_to_selection_sequence_index
#     # }
#
#
# def batch_tree_structure(batch):
#     tree_sizes = [b.structure.is_nodes.shape[0] for b in batch]
#     batched_edge_labels = torch.cat([b.structure.edge_labels for b in batch])
#     batched_is_nodes = torch.cat([b.structure.is_nodes for b in batch])
#     batched_node_order = torch.cat([b.structure.node_order for b in batch])
#     batched_reverse_node_order = torch.cat([b.structure.reverse_node_order for b in batch])
#     batched_edge_order = torch.cat([b.structure.edge_order for b in batch])
#     batched_reverse_edge_order = torch.cat([b.structure.reverse_edge_order for b in batch])
#     batched_adjacency_list = []
#     batched_reverse_adjacency_list = []
#     offset = 0
#     for n, b in zip(tree_sizes, batch):
#         batched_adjacency_list.append(b.structure.adjacency_list + offset)
#         batched_reverse_adjacency_list.append(b.structure.reverse_adjacency_list + offset)
#         offset += n
#     batched_adjacency_list = torch.cat(batched_adjacency_list)
#     batched_reverse_adjacency_list = torch.cat(batched_reverse_adjacency_list)
#     tree_structure = TreeStructure(
#         edge_labels=batched_edge_labels,
#         is_nodes=batched_is_nodes,
#         node_order=batched_node_order,
#         reverse_node_order=batched_reverse_node_order,
#         edge_order=batched_edge_order,
#         reverse_edge_order=batched_reverse_edge_order,
#         adjacency_list=batched_adjacency_list,
#         reverse_adjacency_list=batched_reverse_adjacency_list,
#         tree_sizes=tree_sizes
#     )
#     return tree_structure

#
# node_length_to_examples = dict()
# aligned_text_length_to_examples = dict()
#
#
# def log_example(tokens: List[Token], length_to_examples_dict: Dict[int, List[str]]):
#     if len(tokens) >= 10:
#         if len(tokens) not in length_to_examples_dict:
#             length_to_examples_dict[len(tokens)] = []
#         if len(length_to_examples_dict[len(tokens)]) < 10:
#             length_to_examples_dict[len(tokens)].append(" ".join([t.text for t in tokens]))
#
#
# def log_node_example(tokens: List[Token]):
#     log_example(tokens, node_length_to_examples)
#
#
# def log_aligned_text_example(tokens: List[Token]):
#     log_example(tokens, aligned_text_length_to_examples)
#
#
# def print_logged_examples():
#     print("Node examples:")
#     for length, examples in node_length_to_examples.items():
#         print(f"Length {length}:")
#         for example in examples:
#             print(example)
#     print("Aligned text examples:")
#     for length, examples in aligned_text_length_to_examples.items():
#         print(f"Length {length}:")
#         for example in examples:
#             print(example)
#
WORD_NAMESPACE = "tokens"
OP_NAMESPACE = "operations"
SILENT_NAMESPACE = "silent"
