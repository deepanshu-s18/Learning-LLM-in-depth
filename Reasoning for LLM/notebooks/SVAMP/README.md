<h2 align="center">
  SVAMP
</h2>
<h5 align="center"> Are NLP Models really able to Solve Simple Math Word Problems?</h5>

<p align="center">
  <a href="https://2021.naacl.org/"><img src="https://img.shields.io/badge/NAACL-2021-blue"></a>
  <a href="https://arxiv.org/abs/2103.07191"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://github.com/arkilpatel/SVAMP/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>

The task of solving Math Word Problems (MWPs) has received significant research attention in the past years. An MWP consists of a short Natural Language narrative that describes a state of the world and poses a question about some unknown quantities (see Table 1 for examples).

<h2 align="center">
  <img align="center"  src="./images/Table1.png" alt="..." width="350">
</h2>
<p style="text-align: justify;">
In this work, we show deficiencies in two benchmark datasets - <a href="https://github.com/chaochun/nlu-asdiv-dataset">ASDiv-A</a> and <a href="https://github.com/sroy9/mawps">MAWPS</a>. We first show that existing models achieve reasonably high accuracies on these datasets even after removing the "question" part of the MWP at test time. We further show that a simple model without any word-order information can also solve a majority of MWPs in these datasets. Our experiments indicate that existing models rely on shallow heuristics in benchmark MWP datasets for achieving high performance.
</p>

<p style="text-align: justify;">
Our experiments render the benchmark datasets unreliable to measure model performance. To enable more robust evaluation of automatic MWP solvers, we created a challenge set called "SVAMP". The examples in SVAMP test a model across different aspects of solving MWPs. Table 1 provides three examples from SVAMP that test whether a model is Question-sensitive, has robust reasoning ability or is invariant to structural alterations respectively.
</p>


#### Dependencies

- compatible with python 3.6
- dependencies can be installed using `SVAMP/code/requirements.txt`

#### Setup

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages:

at `SVAMP/code:`

```shell
$ pip install -r requirements.txt
```

To create the relevant directories, run the following command in the corresponding directory of that model:

for eg, at `SVAMP/code/graph2tree:`

```shell
$ sh setup.sh
```

Then transfer all the data folders to the data subdirectory of that model. For example, copy the MAWPS data directory i.e. `cv_mawps` from `SVAMP/data` to `SVAMP/code/graph2tree/data/`.

#### Models

The current repository includes 5 implementations of Models:

- RNN Seq2Seq at `SVAMP/code/rnn_seq2seq`
  - Basic Encoder-Decoder with Attention Network. Choice of RNN unit provided among LSTM, GRU or RNN.
- Transformer Seq2Seq at `SVAMP/code/transformer_seq2seq`
  - Basic Transformer Network.
- GTS at `SVAMP/code/gts`
  - RNN Encoder with Tree-based Decoder ([Original Implementation](https://github.com/ShichaoSun/math_seq2tree)).
- Graph2Tree at `SVAMP/code/graph2tree`
  - Graph-based Encoder with Tree-based Decoder ([Original Implementation](https://github.com/2003pro/Graph2Tree)).
- Constrained Model at `SVAMP/code/constrained`
  - Constrained model as described in the paper. Feed-Forward Network maps input embeddings to hidden representations and LSTM Decoder with attention generates the equation.

#### Datasets

We work with the following datasets:

- `mawps`
  - [Paper](https://www.aclweb.org/anthology/N16-1136.pdf) and [Github](https://github.com/sroy9/mawps).
  - `Data Size:` 1921
  - Evaluated by Cross-Validation over 5 splits.
  
- `asdiv-a`
  - [Paper](https://www.aclweb.org/anthology/2020.acl-main.92.pdf) and [Github](https://github.com/chaochun/nlu-asdiv-dataset).
  - `Data Size:` 1217
  - Evaluated by Cross-Validation over 5 splits.
  
- `svamp`
  - `SVAMP/SVAMP.json`  
  - `Data Size:` 1000
  - Complete challenge set to be used for evaluation.

A description of the individual data files in the `SVAMP/data` directory is given below:

- `SVAMP/data/cv_asdiv-a`
  - 5-fold Cross Validation splits of ASDiv-A dataset.

- `SVAMP/data/cv_asdiv-a_without_questions`
  - 5-fold Cross Validation splits of ASDiv-A dataset with questions removed in the test sets.

