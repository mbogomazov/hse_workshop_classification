# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.config import *
from src.models.utils import extract_target, gen_split_data, save_metrics_to_json


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Predict & eval best model from Hyperopt')

    model = joblib.load(hyperopt_best_model_file)

    train = pd.read_pickle(featurized_train_data_pkl)

    train, target = extract_target(train)

    train_data, val_data, train_target, val_target = gen_split_data(
        train, target)

    y_predict = model.predict(val_data)

    save_metrics_to_json(val_target, y_predict, hyperopt_metrics)

    test = pd.read_pickle(featurized_test_data_pkl)

    inference_predict = model.predict(test)
    np.savetxt(hyperopt_inference_predict, inference_predict, delimiter=",")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
