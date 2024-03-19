from dataclasses import dataclass
from typing import Dict

from allennlp.data import TextFieldTensors
from torch import Tensor


@dataclass
class RawInput:
    """
    For convenience: a way to gather together things the forward function will need.
    """
    sentence: TextFieldTensors
    gold_operations: TextFieldTensors
    gold_stack_index_or_silent: Tensor
    gold_is_operation_or_argument: Tensor
    gold_complete_stack_tree: Dict[str, Tensor]

