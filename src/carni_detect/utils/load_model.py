import tensorflow as tf
from carni_detect.config import ModelLoadingConfig


def load_model(config: ModelLoadingConfig = ModelLoadingConfig()) -> tf.keras.Model:
    """Loads a TensorFlow model from the specified path.

    Args:
        config (ModelLoadingConfig, optional): Configuration for model loading. Defaults to ModelLoadingConfig().

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    return tf.keras.models.load_model(config.MODEL_PATH)
