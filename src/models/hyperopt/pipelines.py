from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.compose import *
from src.config import *

real_pipe = Pipeline([
    ('scaler', StandardScaler()),
])

cat_pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocess_pipe = ColumnTransformer(transformers=[
    ('real_cols', real_pipe, REAL_COLS),
    ('cat_cols', cat_pipe, TRAIN_CAT_COLS),
    ('ohe_cols', 'passthrough', OHE_COLS)
])
