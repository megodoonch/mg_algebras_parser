"""
Authors Meaghan and Jonas
"""
import logging
from typing import Dict, Any

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import Attention, Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp_models.generation import SimpleSeq2Seq
from torch import Tensor
from torch.nn import NLLLoss, CrossEntropyLoss, LSTM

from minimalist_parser.convert_mgbank.term2actions import OP_CODE, SILENT_CODE, WORD_CODE, ARG_CODE_OPARGSHIFT, \
    SHIFT_ARG_CODE_OPARGSHIFT, OP_CODE_OPARGSHIFT
from minimalist_parser.shift_reduce.data_reader import INDEX_PADDING_VALUE
from minimalist_parser.shift_reduce.information_classes import RawInput
from minimalist_parser.shift_reduce.seq2seq4shift_reduce import Seq2Seq4ShiftReduce
from minimalist_parser.shift_reduce.shift_reduce_sequence_embedder import (ShiftReduceEmbeddingData,
                                                                           ShiftReduceSequenceEmbedder,
                                                                           extract_operations_tensor_with_negative_padding_from_text_field)
from minimalist_parser.shift_reduce.tree_field import WORD_NAMESPACE
from minimalist_parser.shift_reduce.tree_lstm import TreeLSTM

# this doesn't seem to be working as expected right now
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# # logger.setLevel(logging.INFO)

OPERATIONS_NAMESPACE = "operations"
SILENT_NAMESPACE = "silent"


@Model.register("neural_shift_reduce")
class NeuralShiftReduce(Model):

    def __init__(self, vocab: Vocabulary,
                 num_silent_heads: int,
                 silent_heads_dim: int,
                 seq2seq_hidden_dim: int,
                 dropout_rate: float = 0.0,
                 argument_attention: Attention = None,
                 shift_argument_attention: Attention = None):

        # Initialised AFTER we read in all the data with our data reader, building the vocabulary,
        # which is a map between tokens and indices for each namespace.

        super().__init__(vocab)

        # To embed the input sentence
        sentence_embedding_dim = seq2seq_hidden_dim
        self.node_embedding_raw = \
            Embedding(sentence_embedding_dim, num_embeddings=vocab.get_vocab_size(WORD_NAMESPACE))
        self.shift_reduce_sequence_embedder = ShiftReduceSequenceEmbedder(vocab, seq2seq_hidden_dim,
                                                                          sentence_embedding_dim)
        self._source_embedder = BasicTextFieldEmbedder({WORD_NAMESPACE: self.node_embedding_raw})
        self._encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=sentence_embedding_dim,
                                                            hidden_size=seq2seq_hidden_dim,
                                                            batch_first=True,
                                                            num_layers=2,
                                                            dropout=dropout_rate))
        # the decoder lives in here, encoder is self._encoder, which just encodes sentence.
        # all the magic happens in here!
        self._seq2seq = Seq2Seq4ShiftReduce(vocab, seq2seq_hidden_dim, self._encoder,
                                      target_namespace=WORD_NAMESPACE,  # TODO
                                      target_decoder_layers=2,
                                      target_embedding_dim=seq2seq_hidden_dim,
                                      dropout=dropout_rate, beam_size=1)
        self.silent_choice_vectors = torch.nn.Parameter(torch.randn(num_silent_heads, silent_heads_dim))
        self.argument_attention = argument_attention or AdditiveAttention(vector_dim=silent_heads_dim,
                                                                          matrix_dim=seq2seq_hidden_dim,
                                                                          normalize=False)
        self.shift_argument_attention = shift_argument_attention or AdditiveAttention(vector_dim=seq2seq_hidden_dim,
                                                                          matrix_dim=seq2seq_hidden_dim,
                                                                          normalize=False)
        self.seq2seq_hidden_dim = seq2seq_hidden_dim
        self.operation_layer = torch.nn.Linear(seq2seq_hidden_dim, vocab.get_vocab_size(OPERATIONS_NAMESPACE))

        self.argument_loss_function = CrossEntropyLoss(reduction="sum")  # also used for shift arguments
        self.operation_loss_function = CrossEntropyLoss(reduction="sum")  # TODO check if the reduction is correct

        self.argument_accuracy = CategoricalAccuracy()
        self.shift_argument_accuracy = CategoricalAccuracy()
        self.operation_accuracy = CategoricalAccuracy()


        # for encoding stack items
        # output dimension is the dimension of the arguments we need to choose between.
        # This is the dimension of the silent heads (and of the stack items)
        self.tree_lstm = TreeLSTM(seq2seq_hidden_dim, silent_heads_dim)

        # Input for non-words to the tree LSTM
        self.operation_embedding = torch.nn.Embedding(vocab.get_vocab_size(OPERATIONS_NAMESPACE), seq2seq_hidden_dim)
        self.silent_head_embedding = torch.nn.Embedding(vocab.get_vocab_size(SILENT_NAMESPACE), seq2seq_hidden_dim)

        # this prints a table printing all the trainable and frozen parameter counts of this model, listed by submodule
        count_parameters(self)

    def forward(self,
                sentence: TextFieldTensors,
                gold_operations: TextFieldTensors,  # should include Shift
                gold_argument_choice_index: Tensor,  # not sure yet how to do SHift
                gold_argument_choice_mask: Tensor,
                gold_shift_argument_choice_index: Tensor,
                gold_shift_argument_choice_mask: Tensor,
                gold_is_operation_or_argument_or_shift_argument: Tensor,
                # stack_options: Tensor,  # maybe should include sentence for shift?
                # stack_options_mask: Tensor,
                gold_complete_stack_tree: Dict[str, Tensor],
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
                                                (they are all arguments), so they can be provided as one tensor. This
                                                 does not include Shift arguments. Let's
                                                call the length a. So this has shape b x a.
        @param gold_argument_choice_mask: A mask for gold_argument_choice_index with values True or False
        @param gold_shift_argument_choice_index: Long Tensor. Like gold_argument_choice_index, but for the Shift args.
                                                    Let's call this length c for copy ( = the number of shift
                                                    arguments in the sentence). So this has shape b x c.
        @param gold_shift_argument_choice_mask: A mask for gold_shift_argument_choice_index with values True or False.
        @param gold_is_operation_or_argument_or_shift_argument: Long tensor. This has shape b x (o + a + c).
                                                0 here means that the
                                                output token at this position is an operation (from the gold_operations
                                                tensor), and 1 means that the token at this position is an
                                                argument (from the gold_stack_index_or_silent tensor). 2 means that
                                                we have a Shift argument. Technically, this
                                                tensor could be reconstructed from the gold operations, but just
                                                building it in the data reader and passing it along here is simpler.
                                                Padded values are -1.
        @param stack_options: Shape b x a x <stack size>, where stack size denotes the options on the stack (excluding
            the constant silent options). Each entry is an index of a node in the complete stack tree in a linear order
            specified by gold_complete_stack_tree.
        @param stack_options_mask: To compensate for different numbers of arguments in the different sentences in
            the batch. Shape b x a.
        @param gold_complete_stack_tree: Contains the batched stack trees. TODO Exact dictionary format TBD.
        @return: A dictionary, with at least an entry mapping "loss" to the loss tensor.
        """


        # raw_input = RawInput(sentence=sentence,
        #                      gold_operations=gold_operations,
        #                      gold_stack_index_or_silent=gold_argument_choice_index,
        #                      gold_is_operation_or_argument=gold_is_operation_or_argument,
        #                      gold_complete_stack_tree=gold_complete_stack_tree,
        #                      )

        # We're using an LSTM (not BART), so the shape is
        # batch size x max sentence length in batch x self.seq2seq_hidden_dim
        encoded_sentence_state = self._seq2seq.encode(sentence)
        encoded_sentence = encoded_sentence_state["encoder_outputs"]
        encoded_sentence_tokens = encoded_sentence

        # we might want this later when we use BART
        # Translate the model-dependent format to a standardized format: shape b x s x hidden_size_s
        # encoded_sentence_tokens = self.seq2seq.get_token_encodings(encoded_sentence)

        node_features = self.compute_node_features(encoded_sentence_tokens, gold_complete_stack_tree)

        # node features need to be in order consistent with adjacency list.
        # mix of embedded tokens, embedded ops, embedded silents.
        encoded_nodes, _ = self.tree_lstm.forward(features=node_features,
                                               node_order=gold_complete_stack_tree["node_order"],  # from data reader
                                               adjacency_list=gold_complete_stack_tree["adjacency_list"],
                                               # from data reader
                                               edge_order=gold_complete_stack_tree["edge_order"]  # from data reader
                                               )

        decoder_embedding_data = ShiftReduceEmbeddingData(
            encoded_sentence_tokens=encoded_sentence_tokens,
            encoded_stack_nodes=encoded_nodes,
            gold_operations=gold_operations,
            gold_argument_choice_index=gold_argument_choice_index,
            gold_shift_argument_choice_index=gold_shift_argument_choice_index,
            gold_is_operation_or_argument_or_shift_argument=gold_is_operation_or_argument_or_shift_argument,
            available_stack_nodes=gold_complete_stack_tree["available_stack_nodes"],
        )

        encoded_sr_sequence = self.shift_reduce_sequence_embedder.forward(decoder_embedding_data,
                                                                          self.silent_choice_vectors)

        # We're training, so we can encode the whole gold output to be the hidden state that will be used
        # to compute the query to choose the next output.
        # shape b x (o + a) x hidden_size_decoder
        output_hidden_states = self._seq2seq.forward(encoded_sentence_state, encoded_sr_sequence)

        # seems to work up to here now. Which means the LSTM is successfully running on the gold SR sequence!
        # only thing missing is making the predictions + computing loss/accuracy


        # we wrote most of this already in the beginning
        # may just need to update a bit
        stack_options = gold_complete_stack_tree["available_stack_nodes"]
        loss = torch.zeros(1, dtype=torch.float).to(stack_options.device)
        batch_size = stack_options.shape[0]
        argument_count = stack_options.shape[1]

        # Time to choose a Merge or Move argument. Use attention to choose the best among stack items and silents
        # TODO Only Merge can choose a silent. Should we learn that, like we learn not to double-choose a stack item?
        # not batching the argument part because the stack size keeps changing.
        for b in range(batch_size):
            # shape a x hidden_size_decoder
            output_hidden_states_arguments_only = output_hidden_states[b][
                gold_is_operation_or_argument_or_shift_argument[b] == ARG_CODE_OPARGSHIFT]
            for arg_pos in range(argument_count):
                if gold_argument_choice_index[b, arg_pos] != INDEX_PADDING_VALUE:
                    stack_options_vectors = encoded_nodes[stack_options[b, arg_pos]]
                    all_options_vector = torch.cat([self.silent_choice_vectors, stack_options_vectors], dim=0)
                    attention_scores = self.argument_attention(output_hidden_states_arguments_only[arg_pos].unsqueeze(0),
                                                               all_options_vector.unsqueeze(0))
                    loss += self.argument_loss_function(attention_scores, gold_argument_choice_index[b, arg_pos].unsqueeze(0))
                    self.argument_accuracy(attention_scores, gold_argument_choice_index[b, arg_pos].unsqueeze(0))

        print(f"\n\nloss after arguments {loss.item()}")


        # We know we're shifting. Use attention to choose a word from the sentence.
        # We use the LSTM encodings for each word.
        shift_argument_count = gold_shift_argument_choice_index.shape[1]
        for b in range(batch_size):
            # shape a x hidden_size_decoder
            output_hidden_states_shift_arguments_only = output_hidden_states[b][
                gold_is_operation_or_argument_or_shift_argument[b] == SHIFT_ARG_CODE_OPARGSHIFT]
            for arg_pos in range(shift_argument_count):
                if gold_shift_argument_choice_index[b, arg_pos] != INDEX_PADDING_VALUE:
                    # choose a word from the sentence
                    shift_attention_scores = self.shift_argument_attention(output_hidden_states_shift_arguments_only[arg_pos].unsqueeze(0),
                                                               encoded_sentence_tokens[b])
                    loss += self.argument_loss_function(shift_attention_scores,
                                                        gold_shift_argument_choice_index[b, arg_pos].unsqueeze(0))
                    self.shift_argument_accuracy(shift_attention_scores,
                                                 gold_shift_argument_choice_index[b, arg_pos].unsqueeze(0))
        print(f"loss after shift arguments {loss.item()}")


        # Operation case. These are batched.
        # we flatten the vectors and remove all padding. Could maybe do this more conventionally using a mask
        output_hidden_states_operations_only_flat = get_flat_hidden_states_operations_only(output_hidden_states,
                                                                                 gold_is_operation_or_argument_or_shift_argument)
        operation_scores_flat = self.operation_layer(output_hidden_states_operations_only_flat)
        gold_operations_flat = filter_and_flatten_gold_operations(gold_operations)
        loss += self.operation_loss_function(operation_scores_flat, gold_operations_flat)
        self.operation_accuracy(operation_scores_flat, gold_operations_flat)

        # TODO decoding if we are not training

        return {"loss": loss}

    def compute_node_features(self, encoded_sentence_tokens, gold_complete_stack_tree):
        operation_features = self.operation_embedding(gold_complete_stack_tree["operations"])
        silent_features = self.silent_head_embedding(gold_complete_stack_tree["silent"])
        word_features_list = [encoded_sentence_tokens[s, i]
                              for s, i in gold_complete_stack_tree["word_ids"].cpu().numpy().tolist()]
        word_features = torch.stack(word_features_list, dim=0)
        node_features = torch.zeros([gold_complete_stack_tree["operation_or_word_or_silent"].shape[0],
                                     self.seq2seq_hidden_dim])
        node_features[gold_complete_stack_tree["operation_or_word_or_silent"] == OP_CODE] = operation_features
        node_features[gold_complete_stack_tree["operation_or_word_or_silent"] == SILENT_CODE] = silent_features
        node_features[gold_complete_stack_tree["operation_or_word_or_silent"] == WORD_CODE] = word_features
        return node_features

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "argument_accuracy": self.argument_accuracy.get_metric(reset=reset),
            "shift_argument_accuracy": self.shift_argument_accuracy.get_metric(reset=reset),
            "operation_accuracy": self.operation_accuracy.get_metric(reset=reset),
        }


def get_flat_hidden_states_operations_only(output_hidden_states, gold_is_operation_or_argument_or_shift_argument):
    output_hidden_states_flat = output_hidden_states.contiguous().view(-1, output_hidden_states.shape[-1]) # TODO the need to call contiguous() here is a bit odd. If there is a bug, this may be a reason.
    gold_is_operation_or_argument_or_shift_argument_flat = gold_is_operation_or_argument_or_shift_argument.view(-1)
    return output_hidden_states_flat[gold_is_operation_or_argument_or_shift_argument_flat == OP_CODE_OPARGSHIFT]

def filter_and_flatten_gold_operations(gold_operations):
    tensor_with_negative_padding = extract_operations_tensor_with_negative_padding_from_text_field(gold_operations)
    flattened_gold_operations = tensor_with_negative_padding.view(-1)
    return flattened_gold_operations[flattened_gold_operations != INDEX_PADDING_VALUE]

def count_parameters(model):
    print("Trainable Params:")
    trainable_param_count = count_parameters_trainable_specified(model, True)
    print(f"Total Trainable Params: {trainable_param_count}\n\n")
    print("Frozen Params:")
    non_trainable_param_count = count_parameters_trainable_specified(model, False)
    print(f"Total Frozen Params: {non_trainable_param_count}")


def count_parameters_trainable_specified(model, only_if_requires_grad):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == only_if_requires_grad:
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    return total_params
