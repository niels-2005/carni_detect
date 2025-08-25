import tensorflow as tf
from carni_detect.config import TrainingCallbacksConfig


def get_early_stopping_callback(
    config: TrainingCallbacksConfig,
) -> tf.keras.callbacks.EarlyStopping:
    """
    Creates an EarlyStopping callback.

    Args:
        config (TrainingCallbacksConfig): Configuration for the callback.

    Returns:
        tf.keras.callbacks.EarlyStopping: Configured EarlyStopping callback.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=config.MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )


def get_reduce_lr_callback(
    config: TrainingCallbacksConfig,
) -> tf.keras.callbacks.ReduceLROnPlateau:
    """
    Creates a ReduceLROnPlateau callback.

    Args:
        config (TrainingCallbacksConfig): Configuration for the callback.

    Returns:
        tf.keras.callbacks.ReduceLROnPlateau: Configured ReduceLROnPlateau callback.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=config.MONITOR,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.REDUCE_LR_MIN_LR,
    )


def get_model_checkpoint_callback(
    config: TrainingCallbacksConfig,
) -> tf.keras.callbacks.ModelCheckpoint:
    """
    Creates a ModelCheckpoint callback.

    Args:
        config (TrainingCallbacksConfig): Configuration for the callback.

    Returns:
        tf.keras.callbacks.ModelCheckpoint: Configured ModelCheckpoint callback.
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        save_best_only=True,
        monitor=config.MONITOR,
    )


def get_tensorboard_callback(
    config: TrainingCallbacksConfig,
) -> tf.keras.callbacks.TensorBoard:
    """
    Creates a TensorBoard callback.

    Args:
        config (TrainingCallbacksConfig): Configuration for the callback.

    Returns:
        tf.keras.callbacks.TensorBoard: Configured TensorBoard callback.
    """
    return tf.keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_LOG_DIR)


def get_training_callbacks(
    config: TrainingCallbacksConfig = TrainingCallbacksConfig(),
):
    """
    Creates a list of training callbacks.

    Args:
        config (TrainingCallbacksConfig, optional): Configuration for the callbacks. Defaults to TrainingCallbacksConfig().

    Returns:
        list: List of configured training callbacks.
    """
    return [
        get_early_stopping_callback(config),
        get_model_checkpoint_callback(config),
        get_tensorboard_callback(config),
        get_reduce_lr_callback(config),
    ]
