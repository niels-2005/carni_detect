from src.carni_detect.utils.get_image_datasets import get_training_datasets
from src.carni_detect.utils.build_model import build_model
from src.carni_detect.utils.get_training_callbacks import get_training_callbacks
from src.carni_detect.config import ModelTrainingConfig


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
