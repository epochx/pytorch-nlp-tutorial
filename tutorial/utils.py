import torch
import numpy as np

def pad_list(sequences, dim0_pad=None, dim1_pad=None,
             align_right=False, pad_value=0):
    """
    Receives a list of lists and returns a padded 2d ndarray,
    and a list of lengths.

    sequences: a list of lists. len(sequences) = M, and N is the max
               length of any of the lists contained in sequences.
               e.g.: [[2,45,3,23,54], [12,4,2,2], [4], [45, 12]]

    Returns a numpy ndarray of dimension (M, N) corresponding to the padded
    sequences and a list of the original lengths.

    Returns:
       - out: a torch tensor of dimension (M, N)
       - lengths: a list of ints containing the lengths of each element
                  in sequences

    """

    sequences = [np.asarray(sublist) for sublist in sequences]

    if not dim0_pad:
        dim0_pad = len(sequences)

    if not dim1_pad:
        dim1_pad = max(len(seq) for seq in sequences)

    out = np.full(shape=(dim0_pad, dim1_pad), fill_value=pad_value)

    lengths = []
    for i in range(len(sequences)):
        data_length = len(sequences[i])
        lengths.append(data_length)
        offset = dim1_pad - data_length if align_right else 0
        np.put(out[i], range(offset, offset + data_length), sequences[i])

    lengths = np.array(lengths)

    return out, lengths


class Batch:

    def __init__(self, sequences, lengths, sublengths, masks):
        self.sequences = sequences
        self.lengths = lengths
        self.sublengths = sublengths
        self.masks = masks

    def to_torch_(self, device):
        self.sequences = torch.tensor(self.sequences,
                                      device=device,
                                      dtype=torch.long)

        if self.lengths is not None:
            self.lengths = torch.tensor(self.lengths,
                                        device=device,
                                        dtype=torch.long)

        if self.sublengths is not None:
            self.sublengths = torch.tensor(self.sublengths,
                                           device=device,
                                           dtype=torch.long)
        if self.masks is not None:
            self.masks = torch.tensor(self.masks,
                                      device=device,
                                      dtype=torch.float)


class BatchIterator(object):

    def __init__(self, vocabs, examples, batch_size, batch_builder,
                 shuffle=False, max_len=None):

        self.vocabs = vocabs
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.examples = examples
        self.num_batches = (len(self.examples) + batch_size - 1) // batch_size
        self.batch_builder = batch_builder

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        examples_slice = []
        for i, example in enumerate(self.examples, 1):
            examples_slice.append(example)
            if i > 0 and i % (self.batch_size) == 0:
                yield self.batch_builder(examples_slice, self.vocabs, max_len=self.max_len)
                examples_slice = []

        if examples_slice:
            yield self.batch_builder(examples_slice, self.vocabs, max_len=self.max_len)
