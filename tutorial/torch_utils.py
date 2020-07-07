import torch
import torch.nn as nn
import numpy as np

def mean_pooling(batch_hidden_states, batch_lengths):
    """
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :param batch_lengths: list(batch_size)
    :return:
    """
    batch_lengths = batch_lengths.float()
    batch_lengths = batch_lengths.unsqueeze(1)
    if batch_hidden_states.is_cuda:
        batch_lengths = batch_lengths.cuda()

    pooled_batch = torch.sum(batch_hidden_states, 1)
    pooled_batch = pooled_batch / batch_lengths.expand_as(pooled_batch)

    return pooled_batch


def max_pooling(batch_hidden_states):
    """
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :return:
    """
    pooled_batch, _ = torch.max(batch_hidden_states, 1)
    return pooled_batch


def pack_rnn_input(embedded_sequence_batch, sequence_lengths):
    """
    Prepares the special `PackedSequence` object that can be
    efficiently processed by the `nn.LSTM`.

    :param embedded_sequence_batch: torch.Tensor(seq_len, batch_size)

    :param sequence_lengths: list(batch_size)

    :return:
      - `PackedSequence` object containing our padded batch
      - indices to sort back our sentences to their original order
    """

    sequence_lengths = sequence_lengths.cpu().numpy()

    sorted_sequence_lengths = np.sort(sequence_lengths)[::-1]
    sorted_sequence_lengths = torch.from_numpy(sorted_sequence_lengths.copy())

    idx_sort = np.argsort(-sequence_lengths)
    idx_unsort = np.argsort(idx_sort)

    idx_sort = torch.from_numpy(idx_sort)
    idx_unsort = torch.from_numpy(idx_unsort)

    if embedded_sequence_batch.is_cuda:
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()

    embedded_sequence_batch = embedded_sequence_batch.index_select(0, idx_sort)

    # # go back to ints as requested by torch (will change in torch 0.4)
    # int_sequence_lengths = [int(elem) for elem in sorted_sequence_lengths.tolist()]

    # Handling padding in Recurrent Networks
    packed_rnn_input = nn.utils.rnn.pack_padded_sequence(
        embedded_sequence_batch, sorted_sequence_lengths, batch_first=True
    )

    return packed_rnn_input, idx_unsort

def unpack_rnn_output(packed_rnn_output, indices):
    """
     Recover a regular tensor given a `PackedSequence` as returned
     by  `nn.LSTM`

    :param packed_rnn_output: torch object

    :param indices: Variable(LongTensor) of indices to sort output

    :return:
      - Padded tensor

    """
    encoded_sequence_batch, _ = nn.utils.rnn.pad_packed_sequence(
        packed_rnn_output, batch_first=True
    )

    encoded_sequence_batch = encoded_sequence_batch.index_select(0, indices)

    return encoded_sequence_batch