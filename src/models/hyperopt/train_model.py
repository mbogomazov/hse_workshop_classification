# -*- coding: utf-8 -*-
import joblib
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from src.config import *
from src.models.utils import extract_target
from sklearn.pipeline import *
from src.models.hyperopt.pipelines import *
from src.models.hyperopt.find_best_model import *
from src.models.utils import *
from sklearn.multioutput import MultiOutputClassifier


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Use Hyperopt to find best model and train it')

    train = pd.read_pickle(featurized_train_data_pkl)

    train, target = extract_target(train)

    train_data, val_data, train_target, val_target = gen_split_data(
        train, target)

    model = find_best_model_by_hyperopt(train_data, train_target)

    model_pipe = Pipeline([
        ('preprocess', preprocess_pipe),
        ('model', model)
    ])

    multiout_model_pipe = MultiOutputClassifier(model_pipe, n_jobs=4)
    best_model = multiout_model_pipe.fit(train_data, train_target)
    joblib.dump(best_model, hyperopt_best_model_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
