import numpy as np

from ..utils import pad_list, one_hot
from ..batch import Batch, BatchTuple


class SequenceClassificationBatchBuilder(object):
    def __init__(self, vocabs, max_len=None):
        self.vocabs = vocabs
        self.max_len = max_len

    def __call__(self, examples):

        ids_batch = [int(sentence.index) for sentence in examples]

        src_examples = [
            self.vocabs.src.tokens2indices(sentence.tokens[: self.max_len])
            for sentence in examples
        ]

        tgt_examples = [
            self.vocabs.tgt.token2index[sentence.label] for sentence in examples
        ]

        src_padded, src_lengths = pad_list(
            src_examples, pad_value=self.vocabs.src.PAD.hash
        )

        src_batch_tuple = BatchTuple(src_padded, src_lengths, None, None)

        tgt_batch_tuple = BatchTuple(tgt_examples, None, None, None)

        return Batch(
            indices=ids_batch, src=src_batch_tuple, tgt=tgt_batch_tuple
        )


class LogisticRegressionBatchBuilder(object):
    def __init__(self, vocabs, max_len=None):
        self.vocabs = vocabs
        self.max_len = max_len

    def __call__(self, examples):

        ids_batch = [int(sentence.index) for sentence in examples]

        src_examples = [
            self.vocabs.src.tokens2indices(sentence.tokens[: self.max_len])
            for sentence in examples
        ]

        tgt_examples = [
            self.vocabs.tgt.token2index[sentence.label] for sentence in examples
        ]

        src_examples_one_hot = [
            one_hot(src_example, len(self.vocabs.src))
            for src_example in src_examples
        ]

        src_batch = np.vstack([item.sum(0) for item in src_examples_one_hot])

        tgt_batch = np.asarray(tgt_examples, dtype=np.int64)

        return Batch(indices=ids_batch, src=src_batch, tgt=tgt_batch)

