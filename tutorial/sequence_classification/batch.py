
import numpy as np
import torch

from ..utils import Batch, pad_list

def one_hot(labels, num_classes):
    input_size = len(labels)
    labels = np.array(labels)
    matrix = np.zeros((input_size, num_classes), dtype=np.float32)
    matrix[np.arange(input_size), labels] = 1
    return matrix


class SequenceClassificationBatch(object):

    def __init__(self, examples, vocabs, max_len=None):

        self.ids_batch = [int(sentence.index) for sentence in examples]

        src_examples = [vocabs.src.tokens2indices(sentence.tokens[:max_len])
                        for sentence in examples]

        tgt_examples = [vocabs.tgt.token2index[sentence.label]
                        for sentence in examples]

        src_padded, src_lengths = pad_list(src_examples, pad_value=vocabs.src.PAD.hash)

        self.src_batch = Batch(src_padded, src_lengths, None, None)

        self.tgt_batch = Batch(tgt_examples, None, None, None)

    def to_torch_(self, device):
        self.src_batch.to_torch_(device)
        self.tgt_batch.to_torch_(device)



class LogisticRegressionBatch(object):

    def __init__(self, examples, vocabs, max_len=None):

        self.ids_batch = [int(sentence.index) for sentence in examples]

        src_examples = [vocabs.src.tokens2indices(sentence.tokens[:max_len])
                        for sentence in examples]

        tgt_examples = [vocabs.tgt.token2index[sentence.label]
                        for sentence in examples]

        src_examples_one_hot = [one_hot(src_example, len(vocabs.src))
                                for src_example in src_examples]

        self.src_batch = np.vstack([item.sum(0)
                                    for item in src_examples_one_hot])

        self.tgt_batch = np.asarray(tgt_examples, dtype=np.int64)

    def to_torch_(self, device):
        self.src_batch = torch.from_numpy(self.src_batch).to(device)
        self.tgt_batch = torch.from_numpy(self.tgt_batch).to(device)