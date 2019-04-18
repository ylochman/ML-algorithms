from sklearn.metrics import confusion_matrix
import numpy as np


def dice_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def get_class_stats(target_class, cm):
    tp = cm[target_class, target_class]
    fp = np.sum(cm[:, target_class]) - tp
    fn = np.sum(cm[target_class, :]) - tp
    return (tp, fp, fn)


def score_function(prediction, ground_truth):
    pred_flat = prediction.view(-1)
    gt_flat = ground_truth.view(-1)
    cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1, 2])
    tp, fp, fn = get_class_stats(1, cm)
    kidney_score = dice_score(tp, fp, fn)
    tp, fp, fn = get_class_stats(2, cm)
    tumor_score = dice_score(tp, fp, fn)
    return (kidney_score + tumor_score) / 2
