### Node Classification 

First, make sure to download the datasets as discussed in the datasets README.md

Then, make sure the related requirements are installed, as found in *requirements.txt*, with:

```cmd
pip install -r requirements.txt
```

Finally, you can run **bop_node_classification.py** to run for the BoP + CatBoost model for the node classification task. 
Remember to change the user input values in lines 22-42, according to the dataset/-s you wish to run for.

```cmd
python bop_node_classification.py
```


Running this should give you something along the following lines:

```cmd
$ python bop_node_classification.py

###### AIFB  ######
In train: 58086
# labels in train: 140/176
# labels in test: 36/176
# of unique rels: 90     | # of unique nodes: 8285

For efficiency will focus on 2027 nodes. (The labeled nodes + their 1-hop neighbors)

Hop 2
Sparsity 2-hop: 87.88 % (Time needed for this hop: 0.00 mins)
Hop 3
Sparsity 3-hop: 40.23 % (Time needed for this hop: 0.00 mins)
Hop 4
Sparsity 4-hop: 2.32 % (Time needed for this hop: 0.00 mins)
Total time taken for PAMs creation: 0.01 mins..
Generated PAMS in 0.614 seconds.. Will aggregate BoPs from PAMs.

Hop: 1   Shape: (2027, 8285)     Sparsity: 99.82         Nnz: 30074
Hop: 2   Shape: (2027, 8285)     Sparsity: 87.88         Nnz: 2035266
Hop: 3   Shape: (2027, 8285)     Sparsity: 40.23         Nnz: 10038090
Hop: 4   Shape: (2027, 2027)     Sparsity: 2.32  Nnz: 4013607

Vectorizing with sklearn.

Node Features shape: (2027, 10000).

.... CatBoost training over 5 runs ....

Acc ± 1*std
acc    0.9222222222222222 ± 0.0362177911400147

'''

As seen the time taken to generate k-hop PAMs is usually than a second.

For questions, contact bogas.ko [at] gmail [dot] com.
