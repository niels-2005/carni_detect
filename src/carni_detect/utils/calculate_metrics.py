import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os
from carni_detect.config import ModelEvaluationConfig


def calculate_classes_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    config: ModelEvaluationConfig = ModelEvaluationConfig(),
) -> pd.DataFrame:
    """
    Calculates precision, recall, and f1-score for each class based on the true and predicted labels.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels, same shape as y_true.
        classes (np.ndarray): Array of class labels as strings.

    Returns:
        pd.DataFrame: A DataFrame containing the class names, their corresponding f1-score, precision,
                      and recall. Each row corresponds to a class.
    """
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    # generate pandas dataframe
    df = pd.DataFrame.from_dict(report).transpose().reset_index()
    df.rename(columns={"index": "class_name"}, inplace=True)

    os.makedirs(os.path.dirname(config.CLASSIFICATION_REPORT_SAVE_PATH), exist_ok=True)
    df.to_csv(config.CLASSIFICATION_REPORT_SAVE_PATH, index=False)
