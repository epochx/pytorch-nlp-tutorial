# A batch-wise NLP tutorial with PyTorch

A simple batch-wise NLP tutorial using PyTorch.

## Main requirements

- Python 3
- Pytorch
- conda

## Installation


1. Clone this repo in your home directory (recommended)

   ```bash
   git clone https://github.com/epochx/pytorch-nlp-tutorial
   cd pytorch-nlp-tutorial
   ```

2. Create a conda environment. If you don't have conda installed, I recommend using miniconda. You can then easily create and activate a new conda environment with Python 3.6 by executing:

   ```
   conda create -n tutorial python=3.6
   conda activate tutorial
   ```

3.  Run the installation script

    ```bash
    sh ./install.sh
    ```


## Contents

### sequence-classification

A tutorial for sentence classification using PyTorch. We start by showcasing the PyTorch workflow with a simple Logistic Regression for irony detection on the SemEval 2018 Dataset, using one-hot vectors.

Later, we work on Sentiment Classification over the Large IMDB Movie Review Dataset, using a bidirectional LSTM with word embeddings.

Find the self-contained interactive Jupyter Notebook versions here:
 - For the Logistic Regression:
    -  https://goo.gl/cJn4qz
 - For the LSTM: 
    - https://goo.gl/LB9Z5C

To run the examples in your own machine:

- Download the data (will create `~/data/pytorch-nlp-tutorial`)
  - `sh ./sequence_classification/get_data.sh`
- For the Logistic Regression
  -  `python -m tutorial.sequence_classification.run_log_reg`
- For the LSTM
-  `python -m tutorial.sequence_classification.run_lstm`

### sequence-labeling

A tutorial for sequence labeling using a bidirectional LSTM and Conditonal Random Fields for NER for spanish over the CoNLL-2003 data.

Coming soon...

