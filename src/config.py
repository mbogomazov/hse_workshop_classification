import yaml

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

# Data load paths
train_csv = config['load_data']['data_train_path']
test_csv = config['load_data']['data_test_path']

# Data preprocess cols
TARGET_COLS = config['preprocess_data']['cols']['TARGET_COLS']
ID_COL = config['preprocess_data']['cols']['ID_COL']
EDU_COL = config['preprocess_data']['cols']['EDU_COL']
SEX_COL = config['preprocess_data']['cols']['SEX_COL']
FREQ_COL = config['preprocess_data']['cols']['FREQ_COL']
CAT_COLS = config['preprocess_data']['cols']['CAT_COLS']
OHE_COLS = config['preprocess_data']['cols']['OHE_COLS']
REAL_COLS = config['preprocess_data']['cols']['REAL_COLS']

# Data preprocess filepaths
preprocessed_train_data_pkl = config['preprocess_data']['paths']['preprocessed_train_data_pkl']
preprocessed_target_data_pkl = config['preprocess_data']['paths']['preprocessed_target_data_pkl']
preprocessed_test_data_pkl = config['preprocess_data']['paths']['preprocessed_test_data_pkl']

# Generate features step
WAKE_UP_TIME_COL = config['generate_features']['cols']['WAKE_UP_TIME_COL']
FALL_ASLEEP_COL = config['generate_features']['cols']['FALL_ASLEEP_COL']
SLEEP_BEHAVIORS_COL = config['generate_features']['cols']['SLEEP_BEHAVIORS_COL']
SLEEP_DURATION_COL = config['generate_features']['cols']['SLEEP_DURATION_COL']

featurized_train_data_pkl = config['generate_features']['paths']['featurized_train_data_pkl']
featurized_test_data_pkl = config['generate_features']['paths']['featurized_test_data_pkl']
