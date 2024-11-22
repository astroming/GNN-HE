## GNNHE--Simply Combining GNN and Heuristic Encoding.


About
-----
This repository supports the following work:
### ICLR2023 (rejected): Can GNNs Learn Heuristic Information for Link Prediction?

### the updated version: Can GNNs Learn Link Heuristics? A Concise Review and Evaluation of Link Prediction Methods

In this work, we aim to study the link prediction capability of Graph Neural Networks (GNNs). Our analysis shows that GNNs cannot effectively learn structural information related to the number of common neighbors between two nodes, primarily due to the nature of set-based pooling of the neighborhood aggregation scheme. To verify our analysis, we examine the link prediction performance of an approach that incorporates traditional link heuristics (e.g., Common Neighbors) into the GNN.

We conduct experiments on four link prediction datasets from  Open Graph Benchmark (OGB). Empirical study shows that the link heuristics can boost the prediction performance on sparse graphs. In particular, by only utilizing node embeddings in GCN, the models are able to surpass most previous best results on two dense graphs.

96.76\% Hits@20 on ogbl-ddi,

58.105\% Hits@50 on ogbl-collab,

63.54\% Hits@100 on ogbl-ppa,

88.91\% MRR on ogbl-citation2.

#### The best model settings and training records can be found in the log files in related folders.

Requirements
------------

Latest tested combination: Python 3.8.5 + PyTorch 1.10.0 + PyTorch\_Geometric (cuda 102+PyTorch1.10.0) + OGB 1.3.1.

Install python basic libraries

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

<!-- Usages -->
------
<!--
### Contents description

    main_pred.py, utils.py, models.py -->
