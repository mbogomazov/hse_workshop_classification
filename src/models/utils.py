import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from src.models.hyperopt.pipelines import *
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection import *
from itertools import chain
from src.config import *


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(TARGET_COLS, axis=1), df[TARGET_COLS]
    return df, target


def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
                                shuffle=True,
                                stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from:
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size, train_size=train_size,
                                random_state=random_state, stratify=None, shuffle=shuffle)

    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"

    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    cv = MultilabelStratifiedShuffleSplit(
        test_size=n_test, train_size=n_train, random_state=123)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


def gen_split_data(train, target):
    return multilabel_train_test_split(
        train, target, stratify=target, train_size=0.9)
