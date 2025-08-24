import tensorflow as tf


def get_resnet_as_backone():
    """Loads a pre-trained ResNet50 model as the backbone for feature extraction.

    The ResNet50 model is loaded with ImageNet weights, excluding the top classification layer.
    The model's trainable parameters are frozen.

    Returns:
        tf.keras.Model: A ResNet50 model without the top layer.
    """
    resnet_50 = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    resnet_50.trainable = False
    return resnet_50


def build_model_with_resnet(resnet_50):
    """Builds a classification model using ResNet50 as the backbone.

    The model includes a global average pooling layer and a dense output layer with 15 classes.

    Args:
        resnet_50 (tf.keras.Model): A pre-trained ResNet50 model.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    model = tf.keras.Sequential(
        [
            resnet_50,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(15, activation="softmax"),
        ]
    )
    return model


def compile_model(model):
    """Compiles a Keras model with Adam optimizer and categorical crossentropy loss.

    Args:
        model (tf.keras.Model): The Keras model to compile.

    Returns:
        None: The model is compiled in-place.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    return model


def get_model():
    """Creates and compiles a classification model with ResNet50 as the backbone.

    This function combines the steps of loading ResNet50, building the model architecture,
    and compiling the model.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    resnet_50 = get_resnet_as_backone()
    model = build_model_with_resnet(resnet_50)
    model = compile_model(model)
    return model
