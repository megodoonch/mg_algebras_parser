import warnings
from typing import Dict, List, Tuple, Iterable, Any

import numpy
# from overrides import overrides
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, LSTM

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.common.lazy import Lazy
from allennlp.training.metrics import BLEU, CategoricalAccuracy

from minimalist_parser.shift_reduce.shift_reduce_sequence_embedder import ShiftReduceEmbeddingData, \
    ShiftReduceSequenceEmbedder
from minimalist_parser.shift_reduce.tree_field import WORD_NAMESPACE


@Model.register("seq2seq4shift_reduce")
class Seq2Seq4ShiftReduce(Model):
    """
    Based on simple_seq2seq in allennlp models
    This `SimpleSeq2Seq` class is a `Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.
    # Parameters
    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    beam_search : `BeamSearch`, optional (default = `Lazy(BeamSearch)`)
        This is used to during inference to select the tokens of the decoded output sequence.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : `int`, optional (default = `'source_embedding_dim'`)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    target_pretrain_file : `str`, optional (default = `None`)
        Path to target pretrain embedding files
    target_decoder_layers : `int`, optional (default = `1`)
        Nums of layer for decoder
    attention : `Attention`, optional (default = `None`)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio : `float`, optional (default = `0.`)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015](https://arxiv.org/abs/1506.03099).
    use_bleu : `bool`, optional (default = `True`)
        If True, the BLEU metric will be calculated during validation.
    ngram_weights : `Iterable[float]`, optional (default = `(0.25, 0.25, 0.25, 0.25)`)
        Weights to assign to scores for each ngram size.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int,
        encoder: Seq2SeqEncoder,
        beam_search: Lazy[BeamSearch] or None = None,  # Lazy(BeamSearch, max_steps=1040),  # MF can't seem to get this to work otherwise
        attention: Attention = None,
        target_namespace: str = "tokens",
        target_embedding_dim: int = None,
        scheduled_sampling_ratio: float = 0.0,
        use_bleu: bool = True,
        bleu_ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
        target_pretrain_file: str = None,
        target_decoder_layers: int = 1,
        **kwargs
    ) -> None:
        super().__init__(vocab)

        self._target_namespace = target_namespace
        self._target_decoder_layers = target_decoder_layers
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        # but we don't have an embedder here. Our embedding is complicated.
        # Idea: we just use a learned parameter
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._start_embedding = torch.nn.Parameter(torch.randn(target_embedding_dim), requires_grad=True)

        # MEGO
        self._val_accuracy = CategoricalAccuracy()
        self._train_accuracy = CategoricalAccuracy()
        self._accuracy = CategoricalAccuracy()

        if use_bleu:
            pad_index = self.vocab.get_token_index(
                self.vocab._padding_token, self._target_namespace
            )
            self._bleu = BLEU(
                bleu_ngram_weights,
                exclude_indices={pad_index, self._end_index, self._start_index},
            )
        else:
            self._bleu = None

        # At prediction time, we'll use a beam search to find the best target sequence.
        # For backwards compatibility, check if beam_size or max_decoding_steps were passed in as
        # kwargs. If so, update the BeamSearch object before constructing and raise a DeprecationWarning
        deprecation_warning = (
            "The parameter {} has been deprecated."
            " Provide this parameter as argument to beam_search instead."
        )
        beam_search_extras = {}
        print("kwargs:", kwargs)
        if "beam_size" in kwargs:
            beam_search_extras["beam_size"] = kwargs["beam_size"]
            warnings.warn(deprecation_warning.format("beam_size"), DeprecationWarning)
        if "max_decoding_steps" in kwargs:
            beam_search_extras["max_steps"] = kwargs["max_decoding_steps"]
            warnings.warn(deprecation_warning.format("max_decoding_steps"), DeprecationWarning)
        beam_search = Lazy(BeamSearch, **beam_search_extras)
        self._beam_search = beam_search.construct(
                 end_index=self._end_index, vocab=self.vocab, **beam_search_extras
             )
        # self._beam_search = beam_search.construct(
        #     end_index=self._end_index, vocab=self.vocab, **beam_search_extras
        # )
        print("max steps:", self._beam_search.max_steps)

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        self._source_embedder = Embedding(
            embedding_dim = embedding_dim,
            num_embeddings=vocab.get_vocab_size(namespace=WORD_NAMESPACE)
        )

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        # Dense embedding of vocab words in the target space.
        if not target_pretrain_file:
            self._target_embedder = Embedding(
                num_embeddings=num_classes, embedding_dim=target_embedding_dim
            )
        else:
            self._target_embedder = Embedding(
                embedding_dim=target_embedding_dim,
                pretrained_file=target_pretrain_file,
                vocab_namespace=self._target_namespace,
                vocab=self.vocab,
            )

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        if self._target_decoder_layers > 1:
            self._decoder_cell = LSTM(
                self._decoder_input_dim,
                self._decoder_output_dim,
                self._target_decoder_layers,
            )
        else:
            self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.
        # Parameters
        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.
        step : `int`
            The time step in beam search decoding.

        # Returns
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.
        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self.compute_next_output_and_state(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    # @overrides
    def forward(
        self,  # type: ignore
        state: Dict[str, torch.Tensor],
        embedded_sr_sequence: Tensor,
    ) -> torch.Tensor:

        """
        Make forward pass with decoder logic for producing hidden states for the entire shift reduce sequence.
        # Parameters
        state : `Dict[str, torch.Tensor]`
           The output of this classes encode function. A dictionary describing the encoder state.
        embedded_sr_sequence : `Tensor`
           The embedded gold shift reduce sequence (i.e. the embedded gold output), as embedded by the
           ShiftReduceSequenceEmbedder class.
        # Returns

        The encoded tensor of all the hidden states (last layer only) on the output sequence. This tensor is
        the direct input to the prediction layers during training.

        `torch.Tensor`

        """

        initial_decoder_state = self._init_decoder_state(state)

        return self._forward_loop(initial_decoder_state, embedded_sr_sequence)

    # @overrides
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize predictions.
        This method overrides `Model.make_output_human_readable`, which gets called after `Model.forward`, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for top_k_predictions in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # we want top-k results.
            if len(top_k_predictions.shape) == 1:
                top_k_predictions = [top_k_predictions]

            batch_predicted_tokens = []
            for indices in top_k_predictions:
                indices = list(indices)
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                predicted_tokens = [
                    self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                    for x in indices
                ]
                batch_predicted_tokens.append(predicted_tokens)

            all_predicted_tokens.append(batch_predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def encode(self, source_tokens: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens[WORD_NAMESPACE]["tokens"])
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"],
            state["source_mask"],
            self._encoder.is_bidirectional(),
        )
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )
        if self._target_decoder_layers > 1:
            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_hidden"] = (
                state["decoder_hidden"].unsqueeze(0).repeat(self._target_decoder_layers, 1, 1)
            )

            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_context"] = (
                state["decoder_context"].unsqueeze(0).repeat(self._target_decoder_layers, 1, 1)
            )

        return state

    def _forward_loop(
        self, full_model_state: Dict[str, torch.Tensor], encoded_sr_sequence: Tensor
    ) -> torch.Tensor:
        """
        Make forward pass during training. Takes an initial model state and the encoded gold output sequence.
        Then runs the decoder, using the encoded gold output sequence as input at each step.
        """

        # Contrary to the original seq2seq implementation, we here process everything since there is no EOS token
        #  currently.
        num_decoding_steps = encoded_sr_sequence.shape[1]

        all_hidden_states = []
        for timestep in range(num_decoding_steps):

            if timestep == 0:
                batch_size = encoded_sr_sequence.shape[0]
                # we use a learned parameter as an "embedding of the start symbol"
                decoder_input_embedding = self._start_embedding.view(1, -1).expand(batch_size, -1)
            else:
                decoder_input_embedding = encoded_sr_sequence[:, timestep - 1]  # offset by 1 so that the input is the *previous* prediction

            hidden_state, full_model_state = self.compute_next_output_and_state(decoder_input_embedding, full_model_state)
            all_hidden_states.append(hidden_state)

        return torch.cat(all_hidden_states, dim=0).transpose(0, 1) # put the batch dimension first again

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def compute_next_output_and_state(
        self, embedded_input: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        @param embedded_input: Shape (batch_size x embedding dim)
        @param state: The full state of the model from the last decoder step

        @return: The next output, and the next full model state.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (num_layers, group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (num_layers, group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]


        if self._attention:
            # shape: (group_size, encoder_output_dim)
            if self._target_decoder_layers > 1:
                attended_input = self._prepare_attended_input(
                    decoder_hidden[0], encoder_outputs, source_mask
                )
            else:
                attended_input = self._prepare_attended_input(
                    decoder_hidden, encoder_outputs, source_mask
                )
            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        if self._target_decoder_layers > 1:
            # shape: (1, batch_size, target_embedding_dim)
            decoder_input = decoder_input.unsqueeze(0)

            # shape (decoder_hidden): (num_layers, batch_size, decoder_output_dim)
            # shape (decoder_context): (num_layers, batch_size, decoder_output_dim)
            output, (decoder_hidden, decoder_context) = self._decoder_cell(
                decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
            )
        else:
            # shape (decoder_hidden): (batch_size, decoder_output_dim)
            # shape (decoder_context): (batch_size, decoder_output_dim)
            decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
            )
            output = decoder_hidden

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        return output, state

    def _prepare_attended_input(
        self,
        decoder_hidden_state: torch.LongTensor = None,
        encoder_outputs: torch.LongTensor = None,
        encoder_outputs_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    def _compute_accuracy(self,
        logits: torch.LongTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
    ):
        """
        MEGO
        Compute categorical accuracy.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes categorical accuracy
         while taking the mask into account.
        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        if self.training:
            self._train_accuracy(logits, relevant_targets, relevant_mask)
        else:
            self._val_accuracy(logits, relevant_targets, relevant_mask)
        self._accuracy(logits, relevant_targets, relevant_mask)

    # @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        This seems to make non-loss metrics available for logging. Note loss is in output_dict
        @param reset:
        @return:
        """
        all_metrics: Dict[str, float] = {}
        # register BLEU for validation set only
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))

        # MEGO
        # register accuracy for train and validation sets
        # all_metrics["val_accuracy"] = self._val_accuracy.get_metric(reset=reset)
        # all_metrics["train_accuracy"] = self._train_accuracy.get_metric(reset=reset)
        all_metrics["accuracy"] = self._accuracy.get_metric(reset=reset)

        return all_metrics

    default_predictor = "seq2seq"
