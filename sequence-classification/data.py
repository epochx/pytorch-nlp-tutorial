
import os
from collections import namedtuple

train_path = "aclImdb/train/"  # source data
test_path = "test/imdb_te.csv"  # test data for grade evaluation.


Sentence = namedtuple('Sentence', ['index', 'string', 'label'])


def read_imdb_movie_dataset(dataset_path):

    indices = []
    text = []
    rating = []

    i = 0

    for filename in os.listdir(os.path.join(dataset_path, "pos")):
        file_path = os.path.join(dataset_path, "pos", filename)
        data = open(file_path, 'r', encoding="ISO-8859-1").read()
        indices.append(i)
        text.append(data)
        rating.append(1)
        i = i + 1

    for filename in os.listdir(os.path.join(dataset_path, "neg")):
        file_path = os.path.join(dataset_path, "neg", filename)
        data = open(file_path, 'r', encoding="ISO-8859-1").read()
        indices.append(i)
        text.append(data)
        rating.append(0)
        i = i + 1

    sentences = [ Sentence(index, text, rating)
                  for index, text, rating in zip(indices, text, rating)]

    return sentences




def read_semeval_2013_dataset(filepath):
    sentences = []
    with open(filepath, "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if '\t' in line:
                string, label = line.split('\t')
                if label == 'positive':
                    label = 0
                elif label == 'negative':
                    label = 1
                else:
                    label = 2
            else:
                string = line
                label = None
            sentences.append(Sentence(i, string, label))

    return sentences




