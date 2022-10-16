# -*- coding: utf-8 -*-
import click
import logging
import joblib
import pandas as pd
from sklearn.pipeline import *
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from src.config import *
from src.models.utils import extract_target, gen_split_data


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    model = CatBoostClassifier(
        iterations=catboost_iterations,
        loss_function=catboost_loss_function,
        eval_metric=catboost_eval_metric,
        learning_rate=catboost_learning_rate,
        bootstrap_type=catboost_bootstrap_type,
        boost_from_average=catboost_boost_from_average,
        # param for controling  cat_features
        ctr_leaf_count_limit=catboost_ctr_leaf_count_limit,
        leaf_estimation_iterations=catboost_leaf_estimation_iterations,
        leaf_estimation_method=catboost_leaf_estimation_method,
        cat_features=TRAIN_CAT_COLS)

    train = pd.read_pickle(featurized_train_data_pkl)

    train, target = extract_target(train)

    train_data, val_data, train_target, val_target = gen_split_data(
        train, target)

    model_cast = MultiOutputClassifier(model)

    pipeline_castboost = Pipeline([
        ('model_cast', model_cast)])

    best_model = pipeline_castboost.fit(train_data, train_target)
    joblib.dump(best_model, catboost_best_model_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
