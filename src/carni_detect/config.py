import os
import tensorflow as tf


class ImageDatasetConfig:
    """
    Configuration class for image dataset settings.

    Attributes:
        DATASET_LABEL_MODE (str): Label mode for the dataset (e.g., "categorical").
        DATASET_IMAGE_SIZE (tuple): Target size for resizing images.
        DATASET_BATCH_SIZE (int): Number of images per batch.
        DATASET_SEED (int): Random seed for dataset shuffling.
        DATASET_TRAIN_DIR (str): Directory path for training dataset.
        DATASET_VAL_DIR (str): Directory path for validation dataset.
    """

    DATASET_LABEL_MODE = "categorical"
    DATASET_IMAGE_SIZE = (224, 224)
    DATASET_BATCH_SIZE = 32
    DATASET_SEED = 42

    DATASET_TRAIN_DIR = "datasetv1/train"
    DATASET_VAL_DIR = "datasetv1/val"
    DATASET_EVALUATION_DIR = "datasetv1/test"


class ModelBuildingConfig:
    """
    Configuration class for model settings.

    Attributes:
        RESNET50_INCLUDE_TOP (bool): Whether to include the top layer of the ResNet50 model.
        RESNET50_INPUT_SHAPE (tuple): Input shape for the ResNet50 model.
        RESNET50_LAYERS_TRAINABLE (bool): Whether the ResNet50 layers are trainable.

        MODEL_NUM_OUTPUT_CLASSES (int): Number of output classes for the model.
        MODEL_OUTPUT_ACTIVATION (str): Activation function for the model's output layer.

        MODEL_OPTIMIZER (tf.keras.optimizers.Optimizer): Optimizer used for training the model.
        MODEL_LOSS (tf.keras.losses.Loss): Loss function used for training the model.
        MODEL_METRICS (list): List of metrics to evaluate the model during training.
    """

    RESNET50_INCLUDE_TOP = False
    RESNET50_INPUT_SHAPE = (224, 224, 3)
    RESNET50_LAYERS_TRAINABLE = False

    MODEL_DROPOUT_RATE = 0.5

    MODEL_NUM_OUTPUT_CLASSES = 15
    MODEL_OUTPUT_ACTIVATION = "softmax"

    MODEL_OPTIMIZER = tf.keras.optimizers.Adam()
    MODEL_LOSS = tf.keras.losses.CategoricalCrossentropy()
    MODEL_METRICS = [tf.keras.metrics.CategoricalAccuracy()]


class ModelTrainingConfig:
    """
    Configuration class for model training.

    Attributes:
        EPOCHS (int): Number of epochs to train the model.
    """

    EPOCHS = 50


class TrainingCallbacksConfig:
    """
    Configuration class for training callbacks.

    Attributes:
        MONITOR (str): Metric to monitor for callbacks.
        EARLY_STOPPING_PATIENCE (int): Number of epochs with no improvement after which training will be stopped.
        REDUCE_LR_PATIENCE (int): Number of epochs with no improvement after which learning rate will be reduced.
        REDUCE_LR_FACTOR (float): Factor by which the learning rate will be reduced.
        REDUCE_LR_MIN_LR (float): Minimum learning rate after reduction.
        MODEL_SAVE_PATH (str): Path to save the best model.
        TENSORBOARD_LOG_DIR (str): Directory to save TensorBoard logs.
    """

    MONITOR = "val_loss"

    EARLY_STOPPING_PATIENCE = 6

    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.1
    REDUCE_LR_MIN_LR = 1e-6

    MODEL_SAVE_PATH = "models/carnidetectv1.keras"

    TENSORBOARD_LOG_DIR = "tensorboard_logs"


class ModelLoadingConfig:
    """
    Configuration class for model loading.

    Attributes:
        MODEL_PATH (str): Path to the saved model.
    """

    MODEL_PATH = "models/carnidetectv1.keras"
