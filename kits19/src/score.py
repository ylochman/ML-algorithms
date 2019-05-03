from numba import jit
from sklearn.metrics import confusion_matrix
import numpy as np
import time


@jit(nopython=True)
def dice_score(tp, fp, fn):
    denom = tp + fp + fn
    if denom == 0:
        return None
    return (2 * tp) / (2 * tp + fp + fn)


@jit(nopython=True)
def calculate_metrics(prediction, gt, target):
    tp = np.sum(np.logical_and(prediction == target, gt == target))
    fp = np.sum(np.logical_and(prediction == target, gt != target))
    fn = np.sum(np.logical_and(prediction != target, gt == target))
    return (tp, fp, fn)


@jit(nopython=True)
def score_function_fast(prediction, ground_truth):
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    tp, fp, fn = calculate_metrics(pred_flat, gt_flat, 1)
    kidney_score = dice_score(tp, fp, fn)
    tp, fp, fn = calculate_metrics(pred_flat, gt_flat, 2)
    tumor_score = dice_score(tp, fp, fn)

    if kidney_score is None and tumor_score is None:
        return 1
    elif kidney_score is None:
        return tumor_score
    elif tumor_score is None:
        return kidney_score
    else:
        return (kidney_score + tumor_score) / 2
