import tensorflow as tf
from src.carni_detect.config import ModelBuildingConfig


def build_model(config: ModelBuildingConfig = ModelBuildingConfig()) -> tf.keras.Model:
    """
    Builds and compiles a TensorFlow model using ResNet50 as the backbone.

    Args:
        config (ModelConfig, optional): Configuration for the model. Defaults to ModelConfig().

    Returns:
        tf.keras.Model: A compiled TensorFlow model.

    The model consists of:
        - A ResNet50 backbone with optional trainable layers.
        - A GlobalAveragePooling2D layer for feature aggregation.
        - A Dense output layer with the specified number of classes and activation function.

    The model is compiled with:
        - Optimizer: Defined in the configuration.
        - Loss function: Defined in the configuration.
        - Metrics: Defined in the configuration.
    """
    # Load the ResNet50 model as Backbone
    resnet_50 = tf.keras.applications.ResNet50(
        include_top=config.RESNET50_INCLUDE_TOP,
        input_shape=config.RESNET50_INPUT_SHAPE,
    )
    # freeze resnet layers
    resnet_50.trainable = config.RESNET50_LAYERS_TRAINABLE

    # build model
    model = tf.keras.Sequential(
        [
            resnet_50,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(config.MODEL_DROPOUT_RATE),
            tf.keras.layers.Dense(
                config.MODEL_NUM_OUTPUT_CLASSES,
                activation=config.MODEL_OUTPUT_ACTIVATION,
            ),
        ]
    )
    model.compile(
        optimizer=config.MODEL_OPTIMIZER,
        loss=config.MODEL_LOSS,
        metrics=config.MODEL_METRICS,
    )
    return model
