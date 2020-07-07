import torch.nn as nn

from ..torch_utils import (
    max_pooling,
    mean_pooling,
    pack_rnn_input,
    unpack_rnn_output,
)


class BiLSTM(nn.Module):
    def __init__(
        self,
        embeddings,
        hidden_size,
        num_labels,
        input_dropout=0,
        output_dropout=0,
        bidirectional=True,
        num_layers=2,
        pooling="mean",
    ):

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

        self.lstm = nn.LSTM(
            self.input_size,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
        )

        self.total_hidden_size = (
            self.hidden_size * 2 if self.bidirectional else self.hidden_size
        )

        self.output_layer = nn.Linear(self.total_hidden_size, self.num_labels)

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, src_batch, tgt_batch=None):

        src_sequences = src_batch.sequences
        src_lengths = src_batch.lengths

        embedded_sequence_batch = self.embeddings(src_sequences)
        embedded_sequence_batch = self.input_dropout(embedded_sequence_batch)

        packed_rnn_input, indices = pack_rnn_input(
            embedded_sequence_batch, src_lengths
        )

        rnn_packed_output, _ = self.lstm(packed_rnn_input)
        encoded_sequence_batch = unpack_rnn_output(rnn_packed_output, indices)

        if self.pooling == "mean":
            # batch_size, hidden_x_dirs
            pooled_batch = mean_pooling(encoded_sequence_batch, src_lengths)

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
