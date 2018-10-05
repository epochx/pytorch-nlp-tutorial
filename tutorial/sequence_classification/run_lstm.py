import torch
import tqdm
import torch.nn as nn
import os
import random
from collections import namedtuple

from .data import read_imdb_movie_dataset
from .batch import SequenceClassificationBatch
from ..utils import BatchIterator
from ..vocab import Vocab
from .lstm_model import BiLSTM

HOME_DIR = os.environ['HOME']

data_path =  os.path.join(HOME_DIR,
                          'data',
                          'pytorch-nlp-tutorial',
                          'sequence-classification')

train_path = os.path.join(data_path, 'aclImdb/train')
test_path = os.path.join(data_path, 'aclImdb/test')

device = torch.device('cuda')

epochs = 100
batch_size = 100
hidden_size = 300
log_interval = 10
num_labels = 2
input_dropout = 0.5
output_dropout = 0.5
bidirectional = True
num_layers = 2
min_count = 2
pooling = 'mean'
lr = 0.001
gradient_clipping = 0.25
embedding_size = 300
cuda = True

train_sentences = read_imdb_movie_dataset(train_path)
test_sentences = read_imdb_movie_dataset(test_path)

random.shuffle(train_sentences)
random.shuffle(test_sentences)

src_vocab = Vocab(min_count=0, add_padding=True)
tgt_vocab = Vocab(no_unk=True, add_padding=False)

for sentence in train_sentences:
    src_vocab.add_tokens(sentence.tokens)
    tgt_vocab.add_tokens([sentence.label])

src_vocab.finish()
tgt_vocab.finish()

Vocabs = namedtuple('Vocabs', ['src', 'tgt'])
vocabs = Vocabs(src_vocab, tgt_vocab)

embedddings = nn.Embedding(len(vocabs.src.index2token),
                           embedding_size,
                           padding_idx=vocabs.src.PAD.hash)

model = BiLSTM(embeddings=embedddings,
               hidden_size=hidden_size,
               num_labels=num_labels,
               input_dropout=input_dropout,
               output_dropout=output_dropout,
               bidirectional=bidirectional,
               num_layers=num_layers,
               pooling=pooling)

if cuda:
    model.cuda()

print(model)

train_batches = BatchIterator(vocabs,
                              train_sentences,
                              batch_size,
                              SequenceClassificationBatch)

test_batches = BatchIterator(vocabs,
                             test_sentences,
                             batch_size,
                             SequenceClassificationBatch)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pbar = tqdm.trange(epochs, desc='Training...')

for epoch in pbar:
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0
    i = 0

    model.train()

    for batch in tqdm.tqdm(train_batches, desc="Training"):

        batch.to_torch_(device)

        ids_batch = batch.ids_batch
        src_batch = batch.src_batch
        tgt_batch = batch.tgt_batch

        loss, predictions, logits = model.forward(src_batch,
                                                  tgt_batch=tgt_batch,
                                                  train_embeddings=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       gradient_clipping)

        optimizer.step()
        correct = (predictions == tgt_batch.sequences).long().sum()
        total = tgt_batch.sequences.size(0)
        epoch_correct += correct.item()
        epoch_total += total
        epoch_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            accuracy  = 100 * epoch_correct / epoch_total

            pbar.write('Loss: {}'.format(epoch_loss / log_interval))
            pbar.write('Accuracy: {}'.format(accuracy))
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0

        i += 1

    test_epoch_correct = 0
    test_epoch_total = 0
    test_epoch_loss = 0

    model.eval()

    for batch in tqdm.tqdm(test_batches, desc='Testing'):

        batch.to_torch_(device)

        ids_batch = batch.ids_batch
        src_batch = batch.src_batch
        tgt_batch = batch.tgt_batch

        loss, predictions, logits = model.forward(src_batch,
                                                  tgt_batch=tgt_batch)

        correct = (predictions == tgt_batch.sequences).long().sum()
        total = tgt_batch.sequences.size(0)
        test_epoch_correct += correct.item()
        test_epoch_total += total
        test_epoch_loss += loss.item()

    test_accuracy = 100 * test_epoch_correct / test_epoch_total

    pbar.write('\n---------------------')
    pbar.write('Test Loss: {}'.format(test_epoch_loss / len(test_batches)))
    pbar.write('Test Accuracy: {}'.format(test_accuracy))
    pbar.write('---------------------\n')
