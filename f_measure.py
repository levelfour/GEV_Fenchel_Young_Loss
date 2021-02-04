import logging
from typing import Optional

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from skorch.callbacks import EarlyStopping
import torch

from main import Classifier
from main import LinearClassifier
from main import assert_binary_data
from main import check_if_gev_loss
from main import check_if_validation
from main import get_data
from main import get_loss_class
from main import get_loss_link
from main import get_validation_params
from main import get_mlflow_experiment_id
from main import log_metric
from main import negative_brier_score


logger = logging.getLogger(__name__)


def _seed(s: int) -> Optional[int]:
    return None if s < 0 else s


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    x, y = get_data(cfg)
    assert_binary_data(x, y)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.evaluation.test_ratio, random_state=_seed(cfg.seed))

    mlflow.set_tracking_uri(hydra.utils.get_original_cwd() + '/mlruns')

    experiment_id = get_mlflow_experiment_id(cfg.name)

    with mlflow.start_run(experiment_id=experiment_id):
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

        if check_if_validation(cfg):
            params = get_validation_params(cfg)
            clf = GridSearchCV(
                clf,
                params,
                refit=True,
                cv=cfg.evaluation.n_cv,
                scoring=negative_brier_score,
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
        mlflow.log_param('loss', cfg.loss.name)
        mlflow.log_param('lr', cfg.training.lr)
        mlflow.log_param('max_epochs', cfg.training.max_epochs)
        mlflow.log_param('tol', cfg.training.tol)
        mlflow.log_param('regularization', cfg.training.regularization)
        mlflow.log_param('seed', _seed(cfg.seed))
        if check_if_gev_loss(loss):
            mlflow.log_param('xi', cfg.loss.xi)

        # Step 1: fit the base CPE model
        x_tr, x_vl, y_tr, y_vl = train_test_split(x_tr, y_tr, test_size=0.3)
        clf.fit(x_tr, y_tr)

        # Step 2: search for the best threshold
        threshold_candidates = np.arange(0.05, 1., 0.05)
        validation_scores = [
            f1_score(y_vl, (clf.predict_proba(x_vl)[:, 1] > th).astype(np.uint8))
            for th in threshold_candidates
        ]
        threshold = threshold_candidates[np.argmax(validation_scores)]

        y_prob = clf.predict_proba(x_te)[:, 1]
        y_pred = (y_prob > threshold).astype(np.uint8)
        log_metric('f1_score', f1_score(y_te, y_pred))
        log_metric('roc_auc_score', roc_auc_score(y_te, y_pred))
        log_metric('best_threshold', threshold)


if __name__ == '__main__':
    main()
