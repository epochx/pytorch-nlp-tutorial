import os
from collections import namedtuple

Sentence = namedtuple("Sentence", ["index", "tokens", "label"])


def read_imdb_movie_dataset(dataset_path):

    indices = []
    text = []
    rating = []

    i = 0

    for filename in os.listdir(os.path.join(dataset_path, "pos")):
        file_path = os.path.join(dataset_path, "pos", filename)
        data = open(file_path, "r", encoding="ISO-8859-1").read()
        indices.append(i)
        text.append(data)
        rating.append(1)
        i = i + 1

    for filename in os.listdir(os.path.join(dataset_path, "neg")):
        file_path = os.path.join(dataset_path, "neg", filename)
        data = open(file_path, "r", encoding="ISO-8859-1").read()
        indices.append(i)
        text.append(data)
        rating.append(0)
        i = i + 1

    sentences = [
        Sentence(index, text.split(), rating)
        for index, text, rating in zip(indices, text, rating)
    ]

    return sentences


def read_semeval_2018_task_3_dataset(dataset_file_path):

    sentences = []

    with open(dataset_file_path) as f:
        # skip header
        f.readline()
        for line in f.readlines():
            if line:
                index, label, text = line.strip().split("\t")
                sentence = Sentence(index, text.split(), label)
                sentences.append(sentence)

    return sentences

