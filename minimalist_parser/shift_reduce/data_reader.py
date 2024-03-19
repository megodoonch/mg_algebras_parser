"""
An AllenNLP data reader for the shift-reduce parser
"""
from typing import Iterable, Optional, Dict

import nltk
import torch
from allennlp.data import Instance, Tokenizer, Field, TokenIndexer, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from overrides import overrides

from minimalist_parser.shift_reduce.special_vocabulary import *
from .tree_field import TreeField
from ..minimalism.mgbank_input_codec import node_pattern, leaf_pattern
from ..convert_mgbank.term2actions import SHIFT

WORD_NAMESPACE = "tokens"
OP_NAMESPACE = "operations"
MERGE = "merge"
MOVE = "move"


@DatasetReader.register("shift_reduce")
class ShiftReduceDatasetReader(DatasetReader):
    """
    A DatasetReader for neural minimalist shift-reduce parser.
    Input files are tab-separated and look like this, per line:

    Example:
    """
    def __init__(self,
                 # putting this in the initialiser means you can change tokenisers with the config file.
                 input_tokenizer: Tokenizer = PretrainedTransformerTokenizer(model_name="facebook/bart-base"),
                 input_token_indexer: TokenIndexer = PretrainedTransformerIndexer(model_name="facebook/bart-base",
                                                                                  namespace=WORD_NAMESPACE)
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
        length_data_entry = 5
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

        # corresponding to node_list, 0 if operation, 1 if word index, 2 if silent head.
        operation_or_word_or_silent = sentence_and_shift_reduce_sequence[3].split()

        # [parent child parent child...] indices in node_list
        adjacency_list = sentence_and_shift_reduce_sequence[4].split()

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
        offset = get_number_of_silent_heads()
        for token in shift_reduce_sequence:
            # the choices are silents + stack indices
            # so choice indices for the stack are offset by the number of silent heads
            if is_index(token):
                gold_arguments.append(int(token) + offset)
            elif is_silent_head(token):
                gold_arguments.append(silent_head2id(token))
        fields["gold_argument_choice_index"] = TensorField(torch.tensor(gold_arguments, dtype=torch.long))

        # Because the padding value is 0, this in the padded/batched version has a 1 wherever gold_arguments has a
        #  value, and a 0 where gold_arguments was padded
        fields["gold_argument_choice_mask"] = TensorField(torch.tensor([1 for _ in gold_arguments], dtype=torch.bool),
                                                          padding_value=0)

        # gold_is_operation_or_argument
        gold_is_operation_or_argument = []
        for token in shift_reduce_sequence:
            if is_operation(token):
                gold_is_operation_or_argument.append(1)
            elif is_silent_head(token) or is_index(token):
                gold_is_operation_or_argument.append(0)
            else:
                raise ValueError(f"{token} is neither operation, index nor silent head.")
        fields["gold_is_operation_or_argument"] = TensorField(torch.tensor(gold_is_operation_or_argument,
                                                                           dtype=torch.long),
                                                              # default is 0, but we have 0's!
                                                              padding_value=-1)

        # Term
        tree = TreeField(node_list, adjacency_list, operation_or_word_or_silent)
        print(tree)

        return Instance(fields)


def is_operation(token: str):
    return token.startswith(MERGE) or token.startswith(MOVE) or token.startswith(SHIFT)


def is_silent_head(token: str) -> bool:
    return token in silent_vocabulary


def is_index(token: str) -> bool:
    return token.isdigit()


def is_conjunction(token: str) -> bool:
    return token in conjunctions
