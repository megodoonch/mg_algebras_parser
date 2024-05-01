"""
An AllenNLP data reader for the shift-reduce parser
"""
from collections import Counter
from typing import Iterable, Optional, Dict

import torch
from allennlp.data import Instance, Tokenizer, Field, TokenIndexer, Token, Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from overrides import overrides

from minimalist_parser.shift_reduce.special_vocabulary import *
from minimalist_parser.shift_reduce.tree_field import TreeField, WORD_NAMESPACE, OP_NAMESPACE, SILENT_NAMESPACE
from minimalist_parser.convert_mgbank.term2actions import SHIFT, SHIFT_ARG_CODE_OPARGSHIFT, ARG_CODE_OPARGSHIFT, \
    OP_CODE_OPARGSHIFT

MERGE = "merge"
MOVE = "move"
INDEX_PADDING_VALUE = -1  # for indices, we want to use -1 for padding (rather than 0, which is also a real index)

def is_merge(operation_name):
    return operation_name.startswith(MERGE)

def is_move(operation_name):
    return operation_name.startswith(MOVE)

def is_shift(operation_name):
    return operation_name.startswith(SHIFT)


@DatasetReader.register("shift_reduce")
class ShiftReduceDatasetReader(DatasetReader):
    """
    A DatasetReader for neural minimalist shift-reduce parser.
    Input files are tab-separated and look like this, per line:

    Example:
    """

    def __init__(self,
                 # putting this in the initialiser means you can change tokenisers with the config file.
                 input_tokenizer: Tokenizer = WhitespaceTokenizer(),
                 # PretrainedTransformerTokenizer(model_name="facebook/bart-base"),
                 input_token_indexer: TokenIndexer = SingleIdTokenIndexer(namespace=WORD_NAMESPACE)
                 # PretrainedTransformerIndexer(model_name="facebook/bart-base", namespace=WORD_NAMESPACE)
                 ):
        super().__init__()
        self.input_tokenizer = input_tokenizer
        self.input_token_indexer = input_token_indexer
        self.op_token_indexer = SingleIdTokenIndexer(namespace=OP_NAMESPACE)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads a corpus file from file_path.
        Calls text_to_instance for every entry in the corpus, creating an iterable over all entries.
        @param file_path: str: path to corpus.
        @return:
        """
        with (open(file_path, "r") as f):
            for line in f:
                instance = self.text_to_instance(line)
                if instance is not None:
                    yield instance

    def text_to_instance(self, text: str) -> Optional[Instance]:
        """
        Does the work of getting an AllenNLP instance from a string entry from the corpus.
        Assume that each corpus entry is a list of operations and stack indices in Polish notation.
        sentence \t mg_name i j mv_name k mg_name l [past] ...
        The Instance is essentially a dict from the parameter names in forward to their input values for this instance.
                sentence: TextFieldTensors,
                gold_operations: TextFieldTensors,
                gold_argument_choice_index: Tensor,
                gold_is_operation_or_argument: Tensor,
                stack_options: Tensor,
                stack_options_mask: Tensor,
                gold_complete_stack_tree: Dict[str, Tensor]
        @param text: a line from the corpus.
        @return: Instance for the forward function.
        """
        fields: Dict[str, Field] = {}
        length_data_entry = 7
        # Look like: Sentence	Actions	Tree Indices	Adjacency List	Operations	Arguments
        # remove trailing \n
        text = text.strip()

        sentence_and_shift_reduce_sequence = text.split("\t")
        if len(sentence_and_shift_reduce_sequence) == 0:
            return None
        elif len(sentence_and_shift_reduce_sequence) != length_data_entry:
            raise ValueError(f"Got {len(sentence_and_shift_reduce_sequence)} elements from line {text}."
                             f" Expected {length_data_entry}.")

        sentence = sentence_and_shift_reduce_sequence[0]  # string, words separated by spaces.

        # operation names, silent heads, stack indices, Shift, buffer indices
        shift_reduce_sequence = sentence_and_shift_reduce_sequence[1].split()

        node_list = sentence_and_shift_reduce_sequence[2].split()  # node labels and word indices
        node_number_list = sentence_and_shift_reduce_sequence[5].split()  # node indices, same order as node_list

        # corresponding to node_list, 0 if operation, 1 if word index, 2 if silent head.
        operation_or_word_or_silent = sentence_and_shift_reduce_sequence[3].split()

        # [parent child parent child...] indices in node_list
        adjacency_list = sentence_and_shift_reduce_sequence[4].split()

        # stack choices: in form [n, i, j] [n, i, k, l] ...
        # one list for each parser step.
        # List contains the node numbers of the roots of the subtrees that would currently be on the stack.
        list_of_stack_choice_root_numbers = sentence_and_shift_reduce_sequence[6].split(", ")
        # print(list_of_stack_choice_root_numbers)
        stack_choice_roots = [step.split() for step in list_of_stack_choice_root_numbers]
        # print(f"stack choice roots: {stack_choice_roots}")
        # print(f"node numbers: {node_number_list}")
        # node_number_list has the node numbers of the nodes, in node_list order

        # For each parser step, a list of what's on the stack in the form of a list of indices from the node_list.
        # If i is on the list,
        # that means that the node n at position i in the node_list is the root of a subtree on the stack at this step.
        # If i is in position p on stack_choice_node_list_indices, that means it's the pth stack item at this step.
        stack_choice_node_list_indices = \
            [[node_number_list.index(node_number)
              for node_number in stack_choice_roots[i]] for i in range(len(stack_choice_roots))]
        # print(f"stack choices: {stack_choice_node_list_indices}")

        # we want a stack to choose from each time we might choose from the stack
        # so whenever we have merge, we have two copies of the stack.
        # we also don't need to look at the stack after the last operation
        stack_choice_list_indices_per_argument = []
        i = 0
        for stack in stack_choice_node_list_indices[:-1]:
            label = shift_reduce_sequence[i]
            # print(label)
            if is_merge(label):
                stack_choice_list_indices_per_argument.append(stack)
                stack_choice_list_indices_per_argument.append(stack)
                i += 3
            elif is_move(label):
                stack_choice_list_indices_per_argument.append(stack)
                i += 2
            elif is_shift(label):
                # we don't need to know what the stack looks like for shift
                # (and adding it here would put this vector out of sync with the other argument vectors)
                i += 2
            else:
                pass
        print(f"new stack choices: {stack_choice_list_indices_per_argument}")

        # max_choices = max(len(l) for l in stack_choice_list_indices_per_argument)
        #
        # padding_value = -1
        # stack_choice_tensors = [torch.tensor(l, dtype=torch.long) for l in stack_choice_list_indices_per_argument]
        # # I don't think so
        # for i in range(len(stack_choice_tensors)):
        #     stack_choice_tensors[i] = torch.nn.functional.pad(stack_choice_tensors[i],
        #                                                       (0, max_choices-len(stack_choice_tensors[i])),
        #                                                       value = padding_value)
        # stack_choice_tensors = torch.stack(stack_choice_tensors)
        # fields["available_stack_nodes"] = TensorField(stack_choice_tensors, padding_value=padding_value)

        # (Just keeping the non-tensor version around) note that these aren't tensors, and are not getting batched or padded. This is because we just
        #  loop over them anyway in the forward function, at the moment. This may turn out to be a cause
        #  of inefficiency in the code.
        # fields["available_stack_nodes"] = MetadataField(metadata=stack_choice_list_indices_per_argument)




        # sentence
        tokenized_sentence = self.input_tokenizer.tokenize(sentence)
        fields["sentence"] = TextField(tokenized_sentence, {WORD_NAMESPACE: self.input_token_indexer})

        # operations
        op_sequence = [token for token in shift_reduce_sequence if is_operation(token)]
        op_token_sequence = [Token(token) for token in op_sequence]
        fields["gold_operations"] = TextField(op_token_sequence, {OP_NAMESPACE: self.op_token_indexer})
        # TODO this does not have a mask built in at the moment


        # gold_argument_choice_index
        gold_arguments = []
        shift_arguments = []
        gold_is_operation_or_argument_or_shift_argument = []
        offset = get_number_of_silent_heads()
        for i, token in enumerate(shift_reduce_sequence):
            # the choices are silents + stack indices
            # so choice indices for the stack are offset by the number of silent heads
            if is_index(token):
                if shift_reduce_sequence[i - 1] == SHIFT:
                    shift_arguments.append(int(token))  # no offset here because this is just a sentence index
                    gold_is_operation_or_argument_or_shift_argument.append(SHIFT_ARG_CODE_OPARGSHIFT)
                else:
                    gold_arguments.append(int(token) + offset)
                    gold_is_operation_or_argument_or_shift_argument.append(ARG_CODE_OPARGSHIFT)
            elif is_silent_head(token):
                gold_arguments.append(silent_head2id(token))
                gold_is_operation_or_argument_or_shift_argument.append(ARG_CODE_OPARGSHIFT)
            elif is_operation(token):
                gold_is_operation_or_argument_or_shift_argument.append(OP_CODE_OPARGSHIFT)
            else:
                raise ValueError(f"{token} is neither operation, index nor silent head.")
        fields["gold_argument_choice_index"] = TensorField(torch.tensor(gold_arguments, dtype=torch.long),
                                                           padding_value=INDEX_PADDING_VALUE)
        fields["gold_shift_argument_choice_index"] = TensorField(torch.tensor(shift_arguments, dtype=torch.long),
                                                                 padding_value=INDEX_PADDING_VALUE)

        # Because the padding value is 0, this in the padded/batched version has a 1 wherever gold_arguments has a
        #  value, and a 0 where gold_arguments was padded
        fields["gold_argument_choice_mask"] = TensorField(torch.tensor([1 for _ in gold_arguments], dtype=torch.bool),
                                                          padding_value=0)
        fields["gold_shift_argument_choice_mask"] = (
            TensorField(torch.tensor([1 for _ in shift_arguments], dtype=torch.bool),
                        padding_value=0))  # the mask is 1 where there is no padding, 0 where there is padding

        fields["gold_is_operation_or_argument_or_shift_argument"] = TensorField(
            torch.tensor(gold_is_operation_or_argument_or_shift_argument,
                         dtype=torch.long),
            # default padding is 0, but we have 0's!
            padding_value=INDEX_PADDING_VALUE)

        # Term
        fields["gold_complete_stack_tree"] = TreeField(node_list, adjacency_list, operation_or_word_or_silent,
                                                       stack_choice_list_indices_per_argument)
        # print(fields["gold_complete_stack_tree"])

        return Instance(fields)


def is_operation(token: str):
    return is_move(token) or is_shift(token) or is_merge(token)


def is_silent_head(token: str) -> bool:
    return token in silent_vocabulary


def is_index(token: str) -> bool:
    return token.isdigit()


def is_conjunction(token: str) -> bool:
    return token in conjunctions


def main():
    reader = ShiftReduceDatasetReader()
    data = [instance for instance in reader._read("data/processed/shift_reduce/toy/neural_input.txt")]

    # can we try running the allennlp train thing?

    # counter = {OP_NAMESPACE: Counter(), SILENT_NAMESPACE: Counter()}
    # for instance in data:
    #     instance["gold_complete_stack_tree"].count_vocab_items(counter)
    # vocab = Vocabulary(counter=counter)
    # batch = []
    # for instance in data:
    #     instance["gold_complete_stack_tree"].index(vocab)
    #     padding_lengths = instance["gold_complete_stack_tree"].get_padding_lengths()
    #     batch.append(instance["gold_complete_stack_tree"].as_tensor(padding_lengths))
    # print(data[0]["gold_complete_stack_tree"].batch_tensors(batch))
    # print_vocab_entries(vocab, OP_NAMESPACE)
    # print_vocab_entries(vocab, SILENT_NAMESPACE)


def print_vocab_entries(vocab: Vocabulary, namespace: str):
    print("Vocabulary namespace", namespace)
    for i in range(vocab.get_vocab_size(namespace)):
        print(f"{i}: {vocab.get_token_from_index(i, namespace)}")


if __name__ == "__main__":
    main()
