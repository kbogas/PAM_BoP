### Node Classification Datasets

All of the datasets have been processed and for each we have generated the following 2 files:
- graph.csv
- labels.csv

They consist of:

1. **graph.csv**: (tab-separated) The whole graph in the form of triples (head, rel, tail).
2. **labels.csv**: (tab-separated) A label dictionary for the nodes containint triples of the form (node_id, label, split). The node_id is an integer corresponding to the integers in the graph.csv. The label is the class of the corresponding node and the split is a choice of ["train", "test"], denoting what split the node belongs to.

Please download them from [here](https://owncloud.skel.iit.demokritos.gr/index.php/s/NEf6Oei35yng5Cd) and put them in a folder named "data" in the current directory.
