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

