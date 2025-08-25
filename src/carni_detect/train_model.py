from .utils import (
    get_training_datasets,
    build_model,
    get_training_callbacks,
)
from .config import ModelTrainingConfig


def train_model(config: ModelTrainingConfig = ModelTrainingConfig()) -> None:
    """Train the model using the specified configuration.

    Args:
        config (ModelTrainingConfig, optional): Configuration for model training. Defaults to ModelTrainingConfig().

    Returns:
        None
    """
    # this function builds and compiles the model.
    model = build_model()

    # get loaded and preprocessed datasets
    train_dataset, val_dataset = get_training_datasets()

    # start training!
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=get_training_callbacks(),
    )
