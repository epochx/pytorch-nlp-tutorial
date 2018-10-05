import numpy as np
import torch
import torch.nn as nn


def mean_pooling(batch_hidden_states, batch_lengths):
    '''
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :param batch_lengths: list(batch_size)
    :return:
    '''
    batch_lengths = batch_lengths.float()
    batch_lengths = batch_lengths.unsqueeze(1)
    if batch_hidden_states.is_cuda:
        batch_lengths = batch_lengths.cuda()

    pooled_batch = torch.sum(batch_hidden_states, 1)
    pooled_batch = pooled_batch / batch_lengths.expand_as(pooled_batch)

    return pooled_batch


def max_pooling(batch_hidden_states):
    '''
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :return:
    '''
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

    sequence_lengths = np.array(sequence_lengths)
    sorted_sequence_lengths = np.sort(sequence_lengths)[::-1]

    idx_sort = np.argsort(-sequence_lengths)
    idx_unsort = np.argsort(idx_sort)

    idx_sort = torch.from_numpy(idx_sort)
    idx_unsort = torch.from_numpy(idx_unsort)

    if embedded_sequence_batch.is_cuda:
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()

    embedded_sequence_batch = embedded_sequence_batch.index_select(0, idx_sort)

    # go back to ints as requested by torch (will change in torch 0.4)
    int_sequence_lengths = [int(elem) for elem in sorted_sequence_lengths.tolist()]

    # Handling padding in Recurrent Networks
    packed_rnn_input = \
        nn.utils.rnn.pack_padded_sequence(embedded_sequence_batch,
                                          int_sequence_lengths,
                                          batch_first=True)

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
    encoded_sequence_batch, _ = \
        nn.utils.rnn.pad_packed_sequence(packed_rnn_output,
                                         batch_first=True)

    encoded_sequence_batch = \
        encoded_sequence_batch.index_select(0, indices)

    return encoded_sequence_batch


class BiLSTM(nn.Module):
    def __init__(self,
                 embeddings,
                 hidden_size,
                 num_labels,
                 input_dropout=0,
                 output_dropout=0,
                 bidirectional=True,
                 num_layers=2,
                 pooling='mean'):

        super(BiLSTM, self).__init__()

        self.embeddings = embeddings
        self.pooling = pooling

        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.num_labels = num_labels

        self.hidden_size = hidden_size

        self.input_size = self.embeddings.embedding_dim

        self.lstm = nn.LSTM(self.input_size,
                            hidden_size,
                            bidirectional=bidirectional,
                            num_layers=num_layers,
                            batch_first=True)

        self.total_hidden_size = \
            self.hidden_size * 2 if self.bidirectional else self.hidden_size

        self.output_layer = nn.Linear(self.total_hidden_size, self.num_labels)

        self.loss_function = nn.CrossEntropyLoss()


    def forward(self, src_batch, tgt_batch=None,
                train_embeddings=False):

        src_sequences = src_batch.sequences
        src_lengths = src_batch.lengths

        embedded_sequence_batch = self.embeddings(src_sequences)
        embedded_sequence_batch = self.input_dropout(embedded_sequence_batch)

        packed_rnn_input, indices = pack_rnn_input(embedded_sequence_batch,
                                                   src_lengths)

        rnn_packed_output, _ = self.lstm(packed_rnn_input)
        encoded_sequence_batch = unpack_rnn_output(rnn_packed_output, indices)

        if self.pooling == "mean":
            # batch_size, hidden_x_dirs
            pooled_batch = mean_pooling(encoded_sequence_batch,
                                        src_lengths)

        elif self.pooling == "max":
            # batch_size, hidden_x_dirs
            pooled_batch = max_pooling(encoded_sequence_batch)
        else:
            raise NotImplementedError

        logits = self.output_layer(pooled_batch)
        _, predictions = logits.max(1)

        if tgt_batch is not None:
            targets = tgt_batch.sequences
            loss = self.loss_function(logits, targets)
        else:
            loss = None

        return loss, predictions, logits