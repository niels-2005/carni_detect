import tensorflow as tf
from src.carni_detect.utils.get_image_dataset import get_image_dataset
from src.carni_detect.utils.get_model import get_model
from src.carni_detect.utils.get_training_callbacks import get_training_callbacks


def get_training_setup():
    model = get_model()
    train_dataset = get_image_dataset(
        data_directory="dataset/train", batch_size=32, shuffle=True
    )

    val_dataset = get_image_dataset(
        data_directory="dataset/val", batch_size=32, shuffle=False
    )

    callbacks = get_training_callbacks()
    return model, train_dataset, val_dataset, callbacks


def train_model():
    model, train_dataset, val_dataset, callbacks = get_training_setup()
    model.fit(
        train_dataset, validation_data=val_dataset, epochs=10, callbacks=callbacks
    )


if __name__ == "__main__":
    train_model()
