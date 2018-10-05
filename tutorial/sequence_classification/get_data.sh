#!/usr/bin/env bash

mkdir -p ~/data/pytorch-nlp-tutorial/sequence-classification

cd ~/data/pytorch-nlp-tutorial/sequence-classification

wget http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

wget https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskA.txt
wget https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt