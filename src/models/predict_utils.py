from sklearn.metrics import *
import numpy as np
import json


def save_metrics_to_json(y_true, y_pred, y_pred_proba, file_name, average='micro'):
    metrics = {
        'precision': precision_score(y_true, y_pred, average=average).tolist(),
        'recall': recall_score(y_true, y_pred, average=average).tolist(),
        'f1': f1_score(y_true, y_pred, average=average).tolist(),
        'accuracy': accuracy_score(y_true, y_pred)
    }

    if average == 'samples':
        average = None

    roc_auc_metrics = roc_auc_score(y_true, np.transpose(
        [pred[:, 1] for pred in y_pred_proba]), average=None).tolist()

    metrics['r_a'] = {}
    for ind, val in enumerate(roc_auc_metrics):
        metrics['r_a'][ind] = val

    with open(file_name, 'w') as f:
        f.write(json.dumps(metrics))
