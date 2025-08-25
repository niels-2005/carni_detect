from .utils import (
    get_predictions,
    load_model,
    get_evaluation_dataset,
    get_classes,
    calculate_classes_metrics,
)


def evaluate_model() -> None:
    """
    Evaluates the trained model on a given evaluation dataset and computes metrics for each class.

    This function performs the following steps:
      1. Loads the trained model.
      2. Retrieves the evaluation dataset and corresponding ground truth labels.
      3. Generates predictions using the model.
      4. Obtains the list of class names.
      5. Calculates and reports metrics for each class.

    Returns:
        None
    """
    model = load_model()
    dataset, y_true = get_evaluation_dataset()
    y_pred = get_predictions(model, dataset)
    class_names = get_classes()
    calculate_classes_metrics(y_true, y_pred, class_names.values())
