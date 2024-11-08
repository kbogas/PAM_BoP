# PAM_BoP
From Primes to Paths: Enabling Fast Multi-Relational Graph Analysis

![PAM](https://github.com/user-attachments/assets/6d1ea694-a58d-4f8d-8423-062f22f27819)


Accompanying code (and source of data) for the work on [PAMs](https://github.com/kbogas/PAM) using Bag of Path (BoP) features.

For each task, node classification, relation prediction and graph regression, the corresponding folder contains the scripts needed to reproduce the results presented in the paper.

An [accompanyning demo](http://143.233.226.63:5000/) showcasing how we can use Bag of Paths to embed the [HetioNet](https://het.io/) Knowledge Graph. We show a TSNE 2-d projection of the BoPs feature vectors of pairs from HetioNet, alongside a tool to find similar pairs given specific head-tail entities.

We use publicly available datasets in our experiments. We zipped together the data for convenience for the node classification and relation prediction tasks, while for the graph regression we utilize commonly-used python modules for data loading.

You can also try out directly the framework by installing the [prime_adj](https://pypi.org/project/prime-adj/) package.

To be updated shortly.


For questions, contact bogas.ko [at] gmail [dot] com.