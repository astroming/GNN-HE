# !!!!!!
NOTE: We have hiden several lines in our scripts. They cannot be executed currently! Instead, the records of training outputs can confirm our experimental results.
# !!!!!!


## GNN+HE--Combining GNN and Heuristic Encoding.


About
-----
This repository supports the following work:
Combining Representation Learning and Classic Heuristics for Link Prediction.

In this work, we aim to study the Combination of the best of both Representation learning and link Heuristics to boost the performance of link prediction.
ComRH separately encodes link heuristics into trainable embedding vectors and directly combine them with the outputs of representation learning. Also, we propose an effective and scalable Sparse Mask Graph Transformer (SMGT). Theoretically, graph Transformers including SMGT still belong to the family of neighborhood aggregation-based GNNs but their neighborhood contains not only locally linked nodes but also global nodes, which leads to a more expressive power of graph Transformer than classic GNNs. In particular, SMGT uses a mask to restrict the size of neighborhood of each node, which allows the attention to be computed in sparse setting, making SMGT scalable for large graph.

We conduct experiments on four link prediction datasets from  Open Graph Benchmark (OGB). ComRH outperforms the top competitors on four OGB benchmark datasets by large margins, gaining new state-of-the-arts, 

94.4\% Hits@20 on ogbl-ddi, 

58.1\% Hits@50 on ogbl-collab, 

62.5\% Hits@100 on ogbl-ppa, 

88.9\% MRR on ogbl-citation2. 

Further empirical study shows that the link heuristics can boost the prediction performance on sparse graphs and representation learning contributes more on very dense graphs.

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
    






