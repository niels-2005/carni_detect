import tensorflow as tf


def get_training_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=3,
        min_lr=1e-6,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_checkpoint.keras",
        save_best_only=True,
        monitor="val_loss",
    )

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")

    return [early_stopping, model_checkpoint, tensorboard, reduce_lr]
