import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import tqdm

from .data import read_semeval_2018_task_3_dataset
from .batch import LogisticRegressionBatchBuilder
from ..vocab import Vocab
from ..utils import Vocabs
from .log_reg_model import LogisticRegression

np.random.seed(42)
torch.manual_seed(42)

HOME_DIR = os.environ["HOME"]

data_path = os.path.join(
    HOME_DIR, "data", "pytorch-nlp-tutorial", "sequence-classification"
)

train_path = os.path.join(data_path, "SemEval2018-T3-train-taskA.txt")
test_path = os.path.join(data_path, "SemEval2018-T3_gold_test_taskA_emoji.txt")

batch_size = 32
min_count = 0
epochs = 20
max_len = 300
learning_rate = 0.5
log_interval = 100
device = torch.device("cuda")

train_examples = read_semeval_2018_task_3_dataset(train_path)
test_examples = read_semeval_2018_task_3_dataset(test_path)

src_vocab = Vocab(min_count=0, add_padding=False)
tgt_vocab = Vocab(no_unk=True, add_padding=False)

for sentence in train_examples:
    src_vocab.add_tokens(sentence.tokens[:max_len])
    tgt_vocab.add_tokens([sentence.label])

src_vocab.finish()
tgt_vocab.finish()

vocabs = Vocabs(src_vocab, tgt_vocab)

batch_builder = LogisticRegressionBatchBuilder(vocabs, max_len=max_len)

train_batches = DataLoader(
    train_examples,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=batch_builder,
)

test_batches = DataLoader(
    test_examples,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=batch_builder,
)

input_size = len(src_vocab)
num_classes = len(tgt_vocab)

model = LogisticRegression(input_size, num_classes)
model = model.to(device=device)
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

pbar = tqdm.trange(epochs)

for epoch in pbar:

    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0
    i = 0

    model.train()

    for train_batch in tqdm.tqdm(train_batches, desc="Training..."):

        train_batch.to_torch_(device)

        optimizer.zero_grad()

        outputs = model(train_batch.src)

        loss = loss_function(outputs, train_batch.tgt)
        loss.backward()

        optimizer.step()

        _, predictions = outputs.max(1)

        correct = (predictions == train_batch.tgt).long().sum()
        total = train_batch.tgt.size(0)
        epoch_correct += correct.item()
        epoch_total += total
        epoch_loss += loss.item()
        i += 1

    accuracy = 100 * epoch_correct / epoch_total

    pbar.write("Epoch {}".format(epoch))
    pbar.write("Train Loss: {}".format(epoch_loss / len(train_batches)))
    pbar.write("Train Accuracy: {}".format(accuracy))

    test_epoch_correct = 0
    test_epoch_total = 0
    test_epoch_loss = 0

    model.eval()

    for test_batch in tqdm.tqdm(test_batches, desc="Evaluating..."):

        test_batch.to_torch_(device)

        outputs = model(test_batch.src)

        loss = loss_function(outputs, test_batch.tgt)

        _, predictions = outputs.max(1)

        correct = (predictions == test_batch.tgt).long().sum()
        total = test_batch.tgt.size(0)
        test_epoch_correct += correct.item()
        test_epoch_total += total
        test_epoch_loss += loss.item()

    test_accuracy = 100 * test_epoch_correct / test_epoch_total

    pbar.write("\n---------------------")
    pbar.write("Test Loss: {}".format(test_epoch_loss / len(test_batches)))
    pbar.write("Test Accuracy: {}".format(test_accuracy))
    pbar.write("---------------------\n")

