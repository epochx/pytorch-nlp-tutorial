import torch
import tqdm
import torch.nn as nn
import os
import random

from .data import read_imdb_movie_dataset
from .utils import Vocab, BatchIterator
from .model import BiLSTM


data_path = '/home/emarrese/data/pytorch-nlp-tutorial/sentence-classification'

train_path = os.path.join(data_path, 'aclImdb/train')
test_path = os.path.join(data_path, 'aclImdb/test')


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



vocab = Vocab(train_sentences,
              min_count=min_count,
              add_padding=True)


embedddings = nn.Embedding(len(vocab.index2token),
                           embedding_size,
                           padding_idx=vocab.PAD.hash)

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

train_batches = BatchIterator(train_sentences,
                              vocab,
                              batch_size,
                              cuda=cuda)

test_batches = BatchIterator(test_sentences,
                             vocab,
                             batch_size,
                             cuda=cuda)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pbar = tqdm.trange(epochs, desc='Training...')

for epoch in pbar:
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0
    for i, batch in enumerate(train_batches):
        (id_sice, padded_x_slice, x_slice_lengths, y_slice) = batch
        loss, predictions, logits = model.forward(padded_x_slice,
                                                  x_slice_lengths,
                                                  y_slice)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(),
                                      gradient_clipping)

        optimizer.step()
        correct = (predictions == y_slice).long().sum()
        total = y_slice.size(0)
        epoch_correct += correct.data[0]
        epoch_total += total
        epoch_loss += loss.data[0]

        if i % log_interval == 0 and i > 0:
            accuracy = 100 * epoch_correct / epoch_total

            pbar.write('Loss: {}'.format(epoch_loss / log_interval))
            pbar.write('Accuracy: {}'.format(accuracy))
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0

    test_epoch_correct = 0
    test_epoch_total = 0
    test_epoch_loss = 0

    for i, batch in enumerate(test_batches):
        (id_sice, padded_x_slice, x_slice_lengths, y_slice) = batch
        loss, predictions, logits = model.forward(padded_x_slice,
                                                  x_slice_lengths,
                                                  y_slice)

        correct = (predictions == y_slice).long().sum()
        total = y_slice.size(0)
        test_epoch_correct += correct.data[0]
        test_epoch_total += total
        test_epoch_loss += loss.data[0]

    test_accuracy = 100 * test_epoch_correct / test_epoch_total

    pbar.write('\n---------------------')
    pbar.write('Loss: {}'.format(test_epoch_loss / len(test_batches)))
    pbar.write('Accuracy: {}'.format(test_accuracy))
    pbar.write('---------------------\n')

