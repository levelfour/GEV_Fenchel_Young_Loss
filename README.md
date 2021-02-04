Implementation of "Fenchel-Young Losses with Skewed Entropies"
==============================================================

This is an official implementation of the following paper:

> Han Bao and Masashi Sugiyama. Fenchel-Young Losses with Skewed Entropies for Class-posterior Probability Estimation. In _AISTATS_, 2021.

The paper provides a convex loss for CPE (class-posterior probability estimation) under class-imbalance,
based on Fenchel-Young losses.

## Requirements

```
pip install -r requirements.txt
```

## Run

### Train a CPE model

```
python main.py loss.name=gev_fenchel_young dataset=test
```

### F-measure maximization based on a CPE model

```
python main.py loss.name=gev_fenchel_young dataset=test
```

### Options

The following methods can be tested (specified for `loss.name`):

+ `gev_fenchel_young`: GEV-Fenchel-Young loss
+ `gev_canonical`: GEV-canonical loss
+ `gev_log`: GEV-log loss
+ `logistic`: logistic regression
+ `hinge`: hinge loss with Platt's scaling
+ `isotonic`: probability calibration with isotonic regression
+ `weight`: balanced logistic regression
+ `bagging`: undersampling with bagging

Please refer to the supplementary material of the paper to see details.

The following datasets are available (specified for `dataset`): `car`, `ecoli`, `glass`, `haberman`, `nursery`, and `yeast`.

More options are available at `config.yaml`.