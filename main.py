from functools import partial
import logging
import os
from typing import Tuple, Optional

import hydra
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import mlflow
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import sklearn.base
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
import torch
from torch import nn

from gev_loss import GEVCanonicalLoss
from gev_loss import GEVFenchelYoungLoss
from gev_loss import GEVLogisticLoss
from gev_loss import gev_inverse_link


np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
logger = logging.getLogger(__name__)


def log_metric(name: str, value: float) -> None:
    logger.info(f"{name}: {value}")
    mlflow.log_metric(name, value)


class Classifier(NeuralNetBinaryClassifier):
    """
    A classifier class adding several tweaks
    into skorch.NeuralNetBinaryClassifier to work well with the newly defined
    loss function classes.
    """

    def decision_function(self, x):
        # necessary for hinge loss used with with CalibratedClassifierCV
        p = self.predict_proba(x)

        """
        Decision function expects 1-dim outputs while BCEWithLogitsLoss with skorch
        uses sigmoid with 2-dim; hence we need to modify outputs into 1-dim.
        """
        if p.ndim == 2:
            p = p[:, 1]

        return p

    def predict(self, x, *arg, **kwarg):
        if self.predict_nonlinearity is None:
            """
            Tweak for hinge loss used with GridSearchCV, which is needed
            because skorch's classifier returns probabilistic outputs,
            while sklearn assumes class label predictions.
            """
            y = self.decision_function(x)
            return (y > 0).astype(np.uint8)
        else:
            return super().predict(x, *arg, **kwarg)

    def set_params(self, *arg, **kwarg):
        """
        Hooked for GEV losses, which is needed
        because the shape parameter of GEV distribution must be the same
        between the loss function and the link function.
        """
        super().set_params(*arg, **kwarg)

        if 'criterion__xi' in kwarg.keys():
            assert self.criterion is not None
            assert check_if_gev_loss(self.criterion)
            xi = kwarg['criterion__xi']
            self.predict_nonlinearity = partial(gev_inverse_link, xi=xi)

        return self


class LinearClassifier(nn.Module):
    def __init__(self, n_features):
        super(LinearClassifier, self).__init__()
        self.f = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.f(x)


class BalancedBaggingRegressor(BaggingRegressor):
    """
    We use BaggingRegressor instead of BaggingClassifier because the latter
    internally uses sklearn.utils.column_or_1d, which changes dtype of y
    from np.float32 to np.int64; eventually this does not work well
    with pytorch.
    """
    def __init__(self,
                 *arg,
                 max_samples=1.0,
                 replacement=False,
                 sampling_strategy='auto',
                 threshold=0.5,
                 **kwarg):
        super(BalancedBaggingRegressor, self).__init__(*arg, **kwarg)
        self.max_samples = max_samples
        self.replacement = replacement
        self.sampling_strategy = sampling_strategy
        self.threshold = threshold

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {}.".format(self.n_estimators))

        if self.base_estimator is not None:
            base_estimator = sklearn.base.clone(self.base_estimator)
        else:
            raise ValueError("base_estimator must be specified.")

        self.base_estimator_ = Pipeline([('sampler', RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            replacement=self.replacement)), ('classifier', base_estimator)])

    def fit(self, x, y):
        return self._fit(x, y, self.max_samples, sample_weight=None)

    def predict(self, x):
        """
        BaggingRegressor.predict is overridden because it calls predict
        of base_estimator, which predicts class labels, we expect.
        """
        p = self.predict_proba(x)[:, 1]
        return (p > self.threshold).astype(np.float32)

    def predict_proba(self, x):
        prob = super(BalancedBaggingRegressor, self).predict(x)
        all_proba = [estim.predict_proba(x) for estim in self.estimators_]
        all_proba = np.array(all_proba)
        proba = all_proba.sum(axis=0) / self.n_estimators
        return proba


class HingeLoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = 2 * target - 1
        return torch.clamp(1 - y * input, 0).mean()


_losses = {
    # name: (loss: torch.nn.Module, inverse_link)
    "bagging": (nn.BCEWithLogitsLoss, 'auto'),
    "gev_canonical": (GEVCanonicalLoss, gev_inverse_link),
    "gev_fenchel_young": (GEVFenchelYoungLoss, gev_inverse_link),
    "gev_log": (GEVLogisticLoss, gev_inverse_link),
    "hinge": (HingeLoss, None),  # Platt's scaling is applied instead of link
    "isotonic": (nn.BCEWithLogitsLoss, 'auto'),
    "logistic": (nn.BCEWithLogitsLoss, 'auto'),
    "weight": (nn.BCEWithLogitsLoss, 'auto'),
}


_gev_losses = [
    GEVCanonicalLoss,
    GEVFenchelYoungLoss,
    GEVLogisticLoss,
]


def get_loss_class(cfg: DictConfig) -> nn.Module:
    return _losses[cfg.loss.name][0]


def get_loss_link(cfg: DictConfig):
    return _losses[cfg.loss.name][1]


def check_if_validation(cfg: DictConfig) -> bool:
    flag = False
    flag |= OmegaConf.is_list(cfg.training.lr)
    flag |= OmegaConf.is_list(cfg.training.regularization)

    if check_if_gev_loss(get_loss_class(cfg)):
        flag |= OmegaConf.is_list(cfg.loss.xi)

    return flag


def check_if_weight(cfg: DictConfig) -> bool:
    return cfg.loss.name == 'weight'


def check_if_gev_loss(loss: torch.nn.Module) -> bool:
    return loss in _gev_losses


def assert_binary_data(x: np.array, y: np.array) -> bool:
    if not np.all((y == 0) | (y == 1)):
        raise ValueError('The specified dataset is not binary.')


def _wrap_with_list(x):
    if OmegaConf.is_list(x):
        return x
    else:
        return [x]


def get_validation_params(cfg: DictConfig) -> dict:
    assert check_if_validation(cfg)
    params = {}

    params['lr'] = _wrap_with_list(cfg.training.lr)
    params['optimizer__weight_decay'] \
        = _wrap_with_list(cfg.training.regularization)

    if check_if_gev_loss(get_loss_class(cfg)):
        params['criterion__xi'] = _wrap_with_list(cfg.loss.xi)

    return params


def get_least_element(xi: np.array):
    elems = np.unique(xi)
    counts = [len(xi[xi == elem]) for elem in elems]
    return elems[np.argmin(counts)]


_openml_datasets = {
    # name: ([openml-id], Optional[positive-label])
    'car': 40975,
    'ecoli': (40671, '4'),
    'glass': (41, 'vehic wind float'),
    'haberman': 43,
    'nursery': 1568,
    'yeast': (181, 'NUC'),
}


def get_data(cfg: DictConfig) -> Tuple[np.array, np.array]:
    if cfg.dataset == 'test':
        n_features = 2
        x, y = make_classification(
            n_samples=1000,
            # features
            n_features=n_features,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            class_sep=2.0,
            # class
            n_classes=2,
            n_clusters_per_class=1,
            # class weight
            weights=(0.5, 0.5),
        )
    else:
        if not cfg.dataset in _openml_datasets.keys():
            raise ValueError(f"dataset `{cfg.dataset}` is not supported")

        info = _openml_datasets[cfg.dataset]
        if isinstance(info, tuple):
            openml_id, positive_label = info
        else:
            openml_id, positive_label = info, None
        dataset = fetch_openml(data_id=openml_id)
        if positive_label is None:
            positive_label = get_least_element(dataset.target)
        x = dataset.data
        y = np.array([1 if _y == positive_label else 0 for _y in dataset.target])

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    logger.info(f"dataset `{cfg.dataset}`: positive_ratio={len(y[y==1])/len(y)}")

    return x, y


def get_mlflow_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name)
    else:
        return experiment.experiment_id


def stratified_brier_score(y_true: np.array, y_prob: np.array, n_bins: int = 10, **kwarg) -> float:
    if len(y_true[y_true == 1]) > 0:
        brier_pos = brier_score_loss(y_true[y_true == 1], y_prob[y_true == 1], **kwarg)
    else:
        brier_pos = np.nan

    if len(y_true[y_true == 0]) > 0:
        brier_neg = brier_score_loss(y_true[y_true == 0], y_prob[y_true == 0], **kwarg)
    else:
        brier_neg = np.nan

    if brier_pos != np.nan and brier_neg != np.nan:
        return (brier_pos + brier_neg) / 2
    elif np.isnan(brier_pos):
        return brier_neg
    elif np.isnan(brier_neg):
        return brier_pos
    else:
        raise RuntimeError()


def _seed(s: int) -> Optional[int]:
    return None if s < 0 else s


def _neg_brier_score(y_true: np.array, y_prob: np.array, **kwarg) -> float:
    """
    To avoid raising exceptions during evaluation of child estimators
    in GridSearchCV. PyTorch returns an estimator regardless of exceptions.
    """
    if np.any(np.isnan(y_prob)):
        return np.nan

    return -stratified_brier_score(y_true, y_prob, **kwarg)


negative_brier_score = make_scorer(
    _neg_brier_score,
    greater_is_better=True,
    needs_proba=True)


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    x, y = get_data(cfg)
    assert_binary_data(x, y)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.evaluation.test_ratio, random_state=_seed(cfg.seed))

    mlflow.set_tracking_uri(hydra.utils.get_original_cwd() + '/mlruns')

    experiment_id = get_mlflow_experiment_id(cfg.name)

    with mlflow.start_run(experiment_id=experiment_id):
        positive_ratio = len(y[y == 1]) / len(y)
        loss = get_loss_class(cfg)
        stopping_criterion = EarlyStopping(
            monitor='train_loss',
            lower_is_better=True,
            patience=min(10, cfg.training.max_epochs),
            threshold=cfg.training.tol,
            threshold_mode='rel',
            sink=logger.info,
        )
        clf = Classifier(
            module=LinearClassifier,
            module__n_features=x.shape[1],
            max_epochs=cfg.training.max_epochs,
            criterion=loss,
            predict_nonlinearity=get_loss_link(cfg),
            optimizer=torch.optim.Adam,
            iterator_train__batch_size=cfg.training.batch_size,
            iterator_train__shuffle=True,
            train_split=False,
            callbacks=[('stopping_criterion', stopping_criterion)],
            verbose=cfg.verbose,
        )

        if check_if_weight(cfg):
            pos_weight = torch.FloatTensor([1 / positive_ratio - 1])
            clf.set_params(criterion__pos_weight=pos_weight)

        if check_if_validation(cfg):
            params = get_validation_params(cfg)
            clf = GridSearchCV(
                clf,
                params,
                refit=True,
                cv=cfg.evaluation.n_cv,
                scoring='accuracy' if loss is HingeLoss else negative_brier_score,
                n_jobs=-1,
            )
        else:
            clf.set_params(
                lr=cfg.training.lr,
                optimizer__weight_decay=cfg.training.regularization,
            )
            if check_if_gev_loss(loss):
                clf.set_params(criterion__xi=cfg.loss.xi)

        mlflow.log_param('dataset', cfg.dataset)
        mlflow.log_param('dataset.positive_ratio', positive_ratio)
        mlflow.log_param('loss', cfg.loss.name)
        mlflow.log_param('lr', cfg.training.lr)
        mlflow.log_param('max_epochs', cfg.training.max_epochs)
        mlflow.log_param('tol', cfg.training.tol)
        mlflow.log_param('regularization', cfg.training.regularization)
        mlflow.log_param('seed', _seed(cfg.seed))
        if check_if_gev_loss(loss):
            mlflow.log_param('xi', cfg.loss.xi)

        if loss is HingeLoss:
            # Step 1: fit linear model with hinge loss
            x_tr, x_vl, y_tr, y_vl = train_test_split(x, y, test_size=0.5)
            clf.fit(x_tr, y_tr)

            # Step 2: calibrate linear model with Platt's scaling
            clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
            clf.fit(x_vl, y_vl)
        elif cfg.loss.name == "isotonic":
            # Step 1: fit linear model with logistic regression (AUC maximization)
            x_tr, x_vl, y_tr, y_vl = train_test_split(x, y, test_size=0.5)
            clf.fit(x_tr, y_tr)

            # Step 2: calibrate linear model with isotonic regression
            clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
            clf.fit(x_vl, y_vl)
        elif cfg.loss.name == "bagging":
            clf = BalancedBaggingRegressor(
                clf,
                n_estimators=10,
                bootstrap=True,
                sampling_strategy='majority',
                n_jobs=1,
            )
            clf.fit(x_tr, y_tr)
        else:
            clf.fit(x_tr, y_tr)

        y_pred = clf.predict(x_te)
        y_prob = clf.predict_proba(x_te)[:, 1]
        log_metric('brier_score', brier_score_loss(y_te, y_prob))


if __name__ == '__main__':
    main()
