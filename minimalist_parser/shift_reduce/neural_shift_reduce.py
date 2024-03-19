"""
Authors Meaghan and Jonas
"""
from typing import Dict, Any

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import Attention
from allennlp.modules.attention import AdditiveAttention
from torch import Tensor
from torch.nn import NLLLoss, CrossEntropyLoss

from minimalist_parser.shift_reduce.information_classes import RawInput
from minimalist_parser.shift_reduce.tree_lstm import TreeLSTM

OPERATIONS_NAMESPACE = "operations"
SILENT_NAMESPACE = "silent"


@Model.register("neural_shift_reduce")
class NeuralShiftReduce(Model):

    def __init__(self, vocab: Vocabulary,
                 num_silent_heads: int,
                 silent_heads_dim: int,
                 seq2seq_hidden_dim: int,
                 argument_attention: Attention = None):

        # Initialised AFTER we read in all the data with our data reader, building the vocabulary,
        # which is a map between tokens and indices for each namespace.

        super().__init__(vocab)
        self.seq2seq = None  # This will implement an interface that we haven't defined yet. See below for what
        self.silent_choice_vectors = torch.nn.Parameter(torch.randn(num_silent_heads, silent_heads_dim))
        self.argument_attention = argument_attention or AdditiveAttention(vector_dim=silent_heads_dim,
                                                                          # TODO this might be wrong
                                                                          matrix_dim=seq2seq_hidden_dim)
        self.seq2seq_hidden_dim = seq2seq_hidden_dim
        self.argument_loss_function = CrossEntropyLoss(reduction="sum")
        self.operation_loss_function = CrossEntropyLoss(reduction="sum")  # TODO check if that's correct
        self.operation_layer = torch.nn.Linear(seq2seq_hidden_dim, vocab.get_vocab_size(OPERATIONS_NAMESPACE))

        # for encoding stack items
        # output dimension is the dimension of the arguments we need to choose between.
        # This is the dimension of the silent heads (and of the stack items)
        self.tree_lstm = TreeLSTM(seq2seq_hidden_dim, silent_heads_dim)

        # Input for non-words to the tree LSTM
        self.operation_embedding = torch.nn.Embedding(vocab.get_vocab_size(OPERATIONS_NAMESPACE), seq2seq_hidden_dim)
        self.silent_head_embedding = torch.nn.Embedding(vocab.get_vocab_size(SILENT_NAMESPACE), seq2seq_hidden_dim)

    def forward(self,
                sentence: TextFieldTensors,
                gold_operations: TextFieldTensors,
                gold_argument_choice_index: Tensor,
                gold_argument_choice_mask: Tensor,
                gold_is_operation_or_argument: Tensor,
                # stack_options: Tensor,
                # stack_options_mask: Tensor,
                # gold_complete_stack_tree: Dict[str, Tensor]
                ) -> Dict[str, Any]:
        """
        The forward pass to predict the next operation, stack index, or silent head. In the following, b denotes the
        batch size.
        The parameters of this function are all output of the DataReader class. Specifically, the DataReader
        provides a dictionary mapping the above parameter names to Field objects (the Field objects are then
        converted to tensors behind the scenes by AllenNLP).
        @param sentence: TextFieldTensor. The full sentence encoded as a TextFieldTensor,
                        which is the default for text. Let's call the length s, so this has shape b x s.
                        TODO These might come with their own masks. If not, we need to add one to the parameters.
        @param gold_operations: TextFieldTensor. Gold operation name sequence. Let's call the length o, so this has
                                    shape b x o.
        @param gold_argument_choice_index: Long Tensor. Both silent elements and stack indices are chosen the same way
                                                (they are all arguments), so they can be provided as one tensor. Let's
                                                call the length a. So this has shape b x a.
        @param gold_argument_choice_mask: A mask for gold_argument_choice_index with values True or False
        @param gold_is_operation_or_argument: Binary tensor. This has shape b x (o + a). "True" here means that the
                                                output token at this position is an operation (from the gold_operations
                                                tensor), and "False" means that the token at this position is an
                                                argument (from the gold_stack_index_or_silent tensor). Technically, this
                                                tensor could be reconstructed from the gold operations, but just
                                                building it in the data reader and passing it along here is simpler.
                                                Actually, "True" is encoded as 1, "False" as 0 and padded values are -1.
        @param stack_options: Shape b x a x <stack size>, where stack size denotes the options on the stack (excluding
            the constant silent options). Each entry is an index of a node in the complete stack tree in a linear order
            specified by gold_complete_stack_tree.
        @param stack_options_mask: To compensate for different numbers of arguments in the different sentences in
            the batch. Shape b x a.
        @param gold_complete_stack_tree: Contains the batched stack trees. TODO Exact dictionary format TBD.
        @return: A dictionary, with at least an entry mapping "loss" to the loss tensor.
        """

        print("sentence:", sentence)
        print("gold_operations:", gold_operations)
        print("gold_argument_choice_index:", gold_argument_choice_index)
        print("gold_argument_choice_mask:", gold_argument_choice_mask)
        print("gold_is_operation_or_argument:", gold_is_operation_or_argument)

        # TODO update to match all function input
        raw_input = RawInput(sentence=sentence,
                             gold_operations=gold_operations,
                             gold_stack_index_or_silent=gold_argument_choice_index,
                             gold_is_operation_or_argument=gold_is_operation_or_argument,
                             gold_complete_stack_tree=None  # gold_complete_stack_tree,
                             )

        # We're using an LSTM (not BART), so the shape is
        # batch size x max sentence length in batch x self.seq2seq_hidden_dim
        encoded_sentence = self.seq2seq.encode(sentence)
        encoded_sentence_tokens = encoded_sentence
        # we might want this later when we use BART
        # Translate the model-dependent format to a standardized format: shape b x s x hidden_size_s
        # encoded_sentence_tokens = self.seq2seq.get_token_encodings(encoded_sentence)

        # shape b x num_nodes x hidden_size_stack (num_nodes = number of nodes in complete stack tree)
        # Probably need hidden_size_stack = hidden_size_s to have compatibility between the vectors for leaves and
        # internal nodes.
        # Computes a vector for each internal node. Output contains vectors for both leaves and internal nodes.
        # The vector is a sequence; the order of nodes is determined by the structure present in
        # gold_complete_stack_tree

        # node features need to be in order consistent with adjacency list.
        # mix of embedded tokens, embedded ops, embedded silents.
        encoded_nodes = self.tree_lstm.forward(features=node_features,
                                               node_order=node_order,  # from data reader
                                               adjacency_list=adjacency_list,  # from data reader
                                               edge_order=edge_order  # from data reader
                                               )

        # We're training, so we can encode the whole gold output to be the hidden state that will be used
        # to compute the query to choose the next output.
        # shape b x (o + a) x hidden_size_decoder
        output_hidden_states = self.seq2seq.encode_gold_output(encoded_sentence, encoded_nodes, raw_input)

        loss = 0
        batch_size = stack_options.shape[0]
        argument_count = stack_options.shape[1]
        # not batching the argument part because the stack size keeps changing.
        for i in range(batch_size):
            encoded_nodes_for_sentence_i = encoded_nodes[i]

            # shape a x hidden_size_decoder
            output_hidden_states_arguments_only = torch.masked_select(output_hidden_states[i],
                                                                      1 - gold_is_operation_or_argument[i])
            for j in range(argument_count):
                if stack_options_mask[i, j] == 1:
                    # TODO the below might need to use a different function, like the select function or such.
                    stack_options_vectors = encoded_nodes_for_sentence_i[stack_options[i, j]]
                    all_options_vector = torch.cat([self.silent_choice_vectors, stack_options_vectors], dim=0)
                    attention_scores = self.argument_attention(output_hidden_states_arguments_only[j],
                                                               all_options_vector)
                    loss += self.argument_loss_function(attention_scores, gold_argument_choice_index)

        # Operation case. These are batched.
        output_hidden_states_operations_only = get_hidden_states_operations_only(output_hidden_states,
                                                                                 gold_is_operation_or_argument)
        operation_scores = self.operation_layer(output_hidden_states_operations_only)
        loss += self.operation_loss_function(operation_scores, gold_operations)


        # TODO decoding if we are not training










