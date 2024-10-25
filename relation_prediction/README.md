### Relation Prediction 

First, make sure to download the datasets as discussed in the data folder README.md

Then, make sure the related requirements are installed, as found in *requirements.txt*, with:

```cmd
pip install -r requirements.txt
```

Finally, you can run **bop_relation_prediction.py** to run for the BoP + kNN for the relation prediction task. 
Remember to change the user input values in lines 22-68, according to the dataset/-s you wish to run for.
The current user selected values are the optimal according to the parameter tuning done.

```cmd
python bop_relation_prediction.py
```

Running this should give you something along the following lines:

```cmd
$ python bop_relation_prediction.py

Running on WN18RR with lossy method...
Dataset details:
In train: 86835
In valid: 3034
In test: 3134
# of unique rels: 11     | # of unique nodes: 40559

Total time for reading: 0.09093 secs (0.00 mins)


Total time for creating PAMs up to P^5: 0.44379 secs (0.01 mins)

Generating train + test features..

Node feats shapes: (40559, 10)

Generated features for the train pairs (86726, 88446).

Generated features for the test pairs (2924, 88446).

Calculate pairwise distances between 2924 (test) pairs and 86726 (train) pairs.

100%|█████████████████████████████████████████████████████████| 89869/89869 [00:00<00:00, 278973.14it/s]
Iterating over test samples and predicting:

 #### RESULTS FOR WN18RR_lossy #######
MRR:0.8164
Hits@1: 0.6850
Hits@3: 0.9463
Hits@10: 0.9590




 Running on WN18RR with lossless method...
Dataset details:
In train: 86835
In valid: 3034
In test: 3134
# of unique rels: 11     | # of unique nodes: 40559

Total time for reading: 0.08941 secs (0.00 mins)


Total time for creating PAMs up to P^2: 8.99536 secs (0.15 mins)

Generating train + test features..

Hop 1, UNQ: 11
Hop 2, UNQ: 94
Total unique paths/features: 105

Node feats shapes: (40559, 210)

Generated features for the train pairs (86726, 630).

Generated features for the test pairs (2924, 630).

Calculate pairwise distances between 2924 (test) pairs and 86726 (train) pairs.

100%|████████████████████████████████████████████████████████████████████| 89869/89869 [00:00<00:00, 211811.66it/s]
Iterating over test samples and predicting:

 #### RESULTS FOR WN18RR_lossless #######
MRR:0.8753
Hits@1: 0.7825
Hits@3: 0.9665
Hits@10: 0.9702



     method dataset       mrr       h@1       h@3      h@10   time_sec
1  lossless  WN18RR  0.875268  0.782490  0.966484  0.970246  36.574776
0     lossy  WN18RR  0.816426  0.685021  0.946306  0.958960  25.056702
```

For questions, contact bogas.ko [at] gmail [dot] com.
