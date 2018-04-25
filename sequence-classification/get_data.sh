#!/usr/bin/env bash

# download and clean enwiki8 dataset (first 2e8 bytes of wiki dumps)
! wget http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
! tar -xzf aclImdb_v1.tar.gz