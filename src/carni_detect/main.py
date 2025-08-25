from src.carni_detect.train_model import train_model
from src.carni_detect.evaluate_model import evaluate_model


def start_pipeline() -> None:
    """
    Starts the training and evaluation pipeline for the model.

    Returns:
        None
    """
    train_model()
    evaluate_model()
