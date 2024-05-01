from dataclasses import dataclass
from typing import List

import allennlp.nn.util
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.nn import Module
from torch import Tensor

from minimalist_parser.shift_reduce.data_reader import INDEX_PADDING_VALUE
from minimalist_parser.shift_reduce.tree_field import OP_NAMESPACE


@dataclass
class ShiftReduceEmbeddingData:
    """
    Some of these are created by the dataset reader.
    The embedded sentence and embedded tree are created in the forward method of the neural shift reduce model.
    These are then compiled into a ShiftReduceEmbeddingData object.
    """
    encoded_sentence_tokens: Tensor
    encoded_stack_nodes: Tensor
    gold_operations: TextFieldTensors  # should include Shift
    gold_argument_choice_index: Tensor
    gold_shift_argument_choice_index: Tensor
    gold_is_operation_or_argument_or_shift_argument: Tensor
    available_stack_nodes: Tensor  # shape batch size x max num args in SR sequence x max available nodes
    # available_stack_nodes contains indices in the flat, concatenated node list (in particular, including offset)


class ShiftReduceSequenceEmbedder(Module):
    def __init__(self,
                 vocab: Vocabulary,
                 embedding_dim: int,
                 sentence_encoding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.operation_embedder = torch.nn.Embedding(vocab.get_vocab_size(OP_NAMESPACE), embedding_dim)
        self.shift_args_linear = torch.nn.Linear(sentence_encoding_dim, embedding_dim)

    def forward(self, embedding_data: ShiftReduceEmbeddingData, silent_embeddings: Tensor):
        """
        Returns the tensor that serves as input to the seq2seq shift-reduce decoder.
        @param embedding_data: ShiftReduceEmbeddingData
        @param silent_embeddings: float Tensor of shape num_silent x hidden_dim (learned model parameter,
        part of a different model, passed to this model as input here)
        @return: Tensor of shape batch size x max shift-reduce sequence length x embedding dim
        TODO the output may need to be offset by one to the right (or maybe the seq2seq model already has that
         logic built in, will need to check)
        """
        # 3 cases: operation, argument (incl silent), shift argument
        # build 3 tensors, put together using gold_is_operation_or_argument_or_shift_argument
        embedded_operations = self.operation_embedder.forward(embedding_data.gold_operations[OP_NAMESPACE]["tokens"])
        shift_arg_shape = embedding_data.gold_shift_argument_choice_index.shape
        embedded_shift_args = torch.zeros(shift_arg_shape[0], shift_arg_shape[1], self.embedding_dim)
        for i, encoded_sentence in enumerate(embedding_data.encoded_sentence_tokens):
            relevant_sentence_embeddings = encoded_sentence[embedding_data.gold_shift_argument_choice_index[i]]
            embedded_shift_args[i] = self.shift_args_linear(relevant_sentence_embeddings)

        available_stack_node_encodings = look_up_available_stack_node_encodings(embedding_data.available_stack_nodes,
                                                                                embedding_data.encoded_stack_nodes)
        embedded_args = extract_selected_argument_vectors_from_available(available_stack_node_encodings,
                                                                         embedding_data.gold_argument_choice_index,
                                                                         silent_embeddings)

        operations_with_negative_padding = extract_operations_tensor_with_negative_padding_from_text_field(
            embedding_data.gold_operations)

        return combine_embeddings(is_operation_or_arg_or_shift_arg=embedding_data.gold_is_operation_or_argument_or_shift_argument,
                                  operation_ids=operations_with_negative_padding,
                                  operations=embedded_operations,
                                  arg_ids=embedding_data.gold_argument_choice_index,
                                  args=embedded_args,
                                  shift_arg_ids=embedding_data.gold_shift_argument_choice_index,
                                  shift_args=embedded_shift_args,
                                  embedding_dim=self.embedding_dim)

def extract_operations_tensor_with_negative_padding_from_text_field(gold_operations: TextFieldTensors):
    # in a text field, the default padding is 0, and it can't be changed in the textfield (as far as I know)
    # so we change the padding manually to -1 here.
    operations_mask = allennlp.nn.util.get_text_field_mask(gold_operations)
    # .clone() should keep the computation path (and we want to make a copy so as to not change the original tensor):
    operations_with_negative_padding = gold_operations[OP_NAMESPACE]["tokens"].clone()
    operations_with_negative_padding[operations_mask == 0] = INDEX_PADDING_VALUE
    return operations_with_negative_padding


def extract_selected_argument_vectors_from_available(available_node_embeddings_tensor,
                                                     gold_argument_choice_index,
                                                     silent_embeddings: Tensor):
    silent_embeddings = silent_embeddings.view(1, 1, silent_embeddings.shape[0], silent_embeddings.shape[1])
    silent_embeddings = silent_embeddings.expand(available_node_embeddings_tensor.shape[0],
                                                 available_node_embeddings_tensor.shape[1],
                                                 -1, -1)
    all_embedding_choices = torch.cat([silent_embeddings, available_node_embeddings_tensor], dim=2)
    # print(all_embedding_choices)
    # print(all_embedding_choices.shape)
    # # flatten everything but the last dimension, so we get a row of vectors
    # all_embedding_choices_flat = all_embedding_choices.view(-1, all_embedding_choices.shape[-1])
    # so the problem is, we can't do lookup across multiple dimensions
    # so we have to flatten the tensors
    # but then the indices are not correct anymore, we would need to offset them
    # maybe we should just use a for loop after all
    batch_size = available_node_embeddings_tensor.shape[0]
    max_num_args_in_sr_sequence = available_node_embeddings_tensor.shape[1]
    results = []
    for i in range(batch_size):
        results.append([])
        for j in range(max_num_args_in_sr_sequence):
            v = all_embedding_choices[i, j, gold_argument_choice_index[i, j]]
            results[i].append(v)
    result_tensors_concat_inside = [torch.cat([v.view(1, -1) for v in l], dim=0) for l in results]
    results_tensor = torch.cat([m.view(1, m.shape[0], m.shape[1]) for m in result_tensors_concat_inside],
                               dim=0)
    return results_tensor


def look_up_available_stack_node_encodings(available_stack_nodes, node_embeddings):
    available_stack_nodes_flat = available_stack_nodes.view(-1)
    # print("available_stack_nodes_flat:", available_stack_nodes_flat)
    # what does this do?
    available_node_embeddings_flat = node_embeddings[available_stack_nodes_flat]
    # print("available_node_embeddings_flat:", available_node_embeddings_flat)
    available_node_embeddings_tensor = available_node_embeddings_flat.view(available_stack_nodes.shape[0],  # batch size
                                                                           available_stack_nodes.shape[1],
                                                                           # num args in SR sequence
                                                                           available_stack_nodes.shape[2],
                                                                           # num available stack nodes (not including silents yet)
                                                                           -1)  # hidden dim
    # this is the red square on our paper, but the red square is only the last two dimensions. This is all batched together.
    return available_node_embeddings_tensor


def combine_embeddings(is_operation_or_arg_or_shift_arg, operation_ids, operations, arg_ids, args, shift_arg_ids,
                       shift_args, embedding_dim):
    # flatten and filter the embeddings
    filtered_flat_operations = flatten_vector_and_remove_pads(operation_ids, operations)
    filtered_flat_args = flatten_vector_and_remove_pads(arg_ids, args)
    filtered_shift_args = flatten_vector_and_remove_pads(shift_arg_ids, shift_args)
    # flatten the is op or arg or shift arg vector
    flat_item_type = is_operation_or_arg_or_shift_arg.view(-1)
    # put everything together in a new (flat for now) result vector
    flat_result = torch.zeros([flat_item_type.shape[0], embedding_dim], dtype=torch.float)
    flat_result[flat_item_type == 0] = filtered_flat_operations
    flat_result[flat_item_type == 1] = filtered_flat_args
    flat_result[flat_item_type == 2] = filtered_shift_args
    # de-flatten
    result = flat_result.view(is_operation_or_arg_or_shift_arg.shape[0], is_operation_or_arg_or_shift_arg.shape[1], embedding_dim)  # maybe need .contiguous()
    return result


def flatten_vector_and_remove_pads(operation_ids, operations):
    flat_operations = operations.view(-1, operations.shape[-1])
    flat_operation_ids = operation_ids.view(-1)
    filtered_flat_operations = flat_operations[flat_operation_ids >= 0]
    return filtered_flat_operations


def main():
    embedding_dim = 10
    tree_sizes_by_sentence = [4, 5, 4]
    total_num_nodes = sum(tree_sizes_by_sentence)  + 1 # +1 for padding
    batch_size = len(tree_sizes_by_sentence)
    # in total 13 nodes. First sentence has 4, second has 5, third has 4

    # sent 0: 3 steps
    # sent 1: 2 steps (padded with -1)
    # sent 2: 3 steps
    gold_argument_choice_index = torch.tensor([[4, 3, 0], [2, 3, -1], [3, 2, 1]], dtype=torch.long)
    print("gold argument choice index:", gold_argument_choice_index)
    # I'm just making up numbers  # doesn't make sense in terms of the SR operations / term, but that's OK

    # node embeddings for the whole forest
    # a.k.a. encoded stack nodes
    node_embeddings = (torch.tensor(range(total_num_nodes), dtype=torch.float)
                       .view(total_num_nodes, 1).expand(total_num_nodes, embedding_dim))

    # print("node_embeddings:", node_embeddings)

    # batch size
    # x max # args in shift-reduce sequence in batch
    # x Max # stack nodes of any given step in any given sent in batch

    # # mg 1 0 mv 3
    # sent0 = [[0,1], [0,1], [3]]
    #
    # # mg 0 1 shift 0 mg 0 4
    # sent1 = [[0,1], [0,1], [0,4], [0,4]]
    #
    # # mg 0 1 mv 3
    # sent2 = [[0,1], [0,1], [3]]
    #
    # max_args = max([len(s) for s in [sent0, sent1, sent2]])
    # max_stack_size = 2
    # offset = 0
    # offset_sentences = []
    # for i, sent in enumerate([sent0, sent1, sent2]):
    #     offset_sentences.append([[n + offset for n in stack_nodes] for stack_nodes in sent])
    #     offset += tree_sizes_by_sentence[i]
    # print("offset sentence", offset_sentences)

    # the arg_ids are offset by the previous tree sizes
    # these probably don't make sense.
    # padding -1
    # max number of steps in this batch: 4
    # sent 0: step 1 only has 3 stack items, step 2 has only 2
    # sent 1: step 0 has 3 stack items; there is no step 2 so it's all -1.
    # Notice these numbers are higher than in step 0 since the indices refer to the batch forest.
    # padding somehow gets an embedding of [13]*10 (13 = total_num_nodes)
    available_stack_nodes = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, -1], [0, 1, -1, -1]],
                             [[4, 5, 6, -1], [4, 5, 7, 8], [-1, -1, -1, -1]],
                             [[10, 11, -1, -1], [10, 11, -1, -1], [9, 10, 11, 12]]], dtype=torch.long)
    # 4 maximum available stack nodes (+ 2 silents, prepended)

    available_node_embeddings_tensor = look_up_available_stack_node_encodings(available_stack_nodes, node_embeddings)

    # print("available_node_embeddings_tensor:", available_node_embeddings_tensor)


    # silents
    num_silents = 2
    silent_embeddings = (torch.tensor([100,200], dtype=torch.float)
                       .view(1, 1, num_silents, 1).expand(available_stack_nodes.shape[0],
                                                          available_stack_nodes.shape[1],
                                                          num_silents, embedding_dim))

    # print("silent embeddings: ", silent_embeddings)

    results_tensor = extract_selected_argument_vectors_from_available(available_node_embeddings_tensor,
                                                                      gold_argument_choice_index, silent_embeddings)

    print(results_tensor)

    # batch_size = 3
    # embedding_dim = 10
    #
    # # create example input data
    # operation_ids = torch.tensor([[5, 8, 3], [2, -1, -1], [3, 4, -1]], dtype=torch.long)
    # arg_ids = torch.tensor([[5, 8, 3], [2, 4, -1], [3, 4, 7]], dtype=torch.long)
    # shift_arg_ids = torch.tensor([[5], [-1], [-1]], dtype=torch.long)
    # operations = (operation_ids * 10).view(batch_size, -1, 1).expand(batch_size, operation_ids.shape[1], embedding_dim).to(torch.float)
    # args = (arg_ids * 100).view(batch_size, -1, 1).expand(batch_size, arg_ids.shape[1], embedding_dim).to(torch.float)
    # shift_args = (shift_arg_ids * 1000).view(batch_size, -1, 1).expand(batch_size, shift_arg_ids.shape[1], embedding_dim).to(torch.float)
    #
    # item_type = torch.tensor([[0, 1, 0, 2, 0, 1, 1], [0, 1, 1, -1, -1, -1, -1], [0, 1, 0, 1, 1, -1, -1]], dtype=torch.long)
    #
    # print("operations", operations)
    # # print("args", args)
    # # print("shift_args", shift_args)
    # # print("item_type", item_type)
    #
    # combine_embeddings(item_type, operation_ids, operations, arg_ids, args, shift_arg_ids, shift_args, embedding_dim)


if __name__ == "__main__":
    main()


