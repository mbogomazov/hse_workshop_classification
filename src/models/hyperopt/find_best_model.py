import pandas as pd
import numpy
from hpsklearn import HyperoptEstimator, one_vs_rest_classifier as one_vs_rest
from hpsklearn import any_preprocessing
from hyperopt import tpe
from src.models.hyperopt.pipelines import *
from sklearn.model_selection import *
from src.config import *


def find_best_model_by_hyperopt(train_data, train_target):

    X = preprocess_pipe.fit_transform(train_data)

    # define search
    model = HyperoptEstimator(classifier=one_vs_rest('clf'),  preprocessing=any_preprocessing(
        'pre'),  algo=tpe.suggest, max_evals=50, trial_timeout=30, seed=seed)
    # perform the search

    model.fit(X, train_target)

    # summarize the best model

    return model.best_model()['learner'].estimator
