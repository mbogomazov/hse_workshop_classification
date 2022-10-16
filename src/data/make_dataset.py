# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
from src.data.preprocess import *
import pandas as pd
from src.config import *


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Preprocess dataset')

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    train = preprocess_data(train)
    test = preprocess_data(test)

    save_as_pickle(train, preprocessed_train_data_pkl)
    save_as_pickle(test, preprocessed_test_data_pkl)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
