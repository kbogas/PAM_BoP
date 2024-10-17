### Graph Regression

Here, we focus on the graph regression task for ZINC, AQSL and Peptides-struct.
These datasets are found in different repos and are shipped as parts of commonly-used libraries which we will utilize for fetching the data.

First, make sure the related requirements are installed, as found in *requirements.txt*, with:

```cmd
pip install -r requirements.txt
```

Then you can run any of the scripts:
- **bop_peptides_struct.py**, for the Peptides-Struct dataset,
- **bop_zinc.py**, for the ZINC (12K) dataset,
- **bop_aqsl.py**, for the AQSL dataset.


Feel free to change the user input values, after the imports, according to the maximum order you wish to run the BoP model for.
The current user selected values are the optimal according to the parameter tuning done.

For example running **bop_peptides_struct.py**, should give you something along the following lines:

```cmd
$ python bop_peptides_struct.py

Start mapping for 15535 graphs + PAM generation @ 6 hops..

100%|████████████████████████████████████████████████████████████████████████████████████| 15535/15535 [00:57<00:00, 271.42it/s]
PAM time taken : 57.90 seconds (0.96 mins)
Generating BoP features..

BoP time took : 37.25 seconds (0.62 mins)
BoP + Nodes (Samples X Feats): (15535, 81146)
Feature selection...
Feature selection took : 15.47 seconds (0.26 mins)

... CatBoostRegressor Training ...

Training took : 100.65 seconds (1.68 mins)
Train MAE: 0.1826
Test MAE: 0.2493
```

For questions, contact bogas.ko [at] gmail [dot] com.
