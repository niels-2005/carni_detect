import numpy as np
import tensorflow as tf


def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tf.Tensor:
    """
    Generates class predictions for a given dataset using a trained TensorFlow model.

    Args:
        model (tf.keras.Model): The trained TensorFlow Keras model used for prediction.
        dataset (tf.data.Dataset): The dataset to predict on, formatted for model input.

    Returns:
        tf.Tensor: A tensor containing the predicted class indices for each sample in the dataset.
    """
    predictions = model.predict(dataset)
    return np.argmax(predictions, axis=1)
