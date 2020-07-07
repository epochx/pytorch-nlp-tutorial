from collections import namedtuple

import numpy as np


Vocabs = namedtuple("Vocabs", ["src", "tgt"])


def one_hot(labels, num_classes):
    input_size = len(labels)
    labels = np.array(labels)
    matrix = np.zeros((input_size, num_classes), dtype=np.float32)
    matrix[np.arange(input_size), labels] = 1
    return matrix


def pad_list(
    sequences, dim0_pad=None, dim1_pad=None, align_right=False, pad_value=0
):
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

