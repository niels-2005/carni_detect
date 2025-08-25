import tensorflow as tf
from src.carni_detect.config import ImageDatasetConfig
import numpy as np


def preprocess_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Preprocesses a TensorFlow dataset by applying ResNet50 preprocessing.

    Args:
        dataset (tf.data.Dataset): The input dataset to preprocess.

    Returns:
        tf.data.Dataset: The preprocessed dataset.
    """
    dataset = dataset.map(
        lambda image, label: (
            tf.keras.applications.resnet50.preprocess_input(image),
            label,
        )
    )
    return dataset


def get_image_dataset(
    config: ImageDatasetConfig, dataset_dir: str, shuffle: bool
) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from the images in the specified directory.

    Args:
        config (ImageDatasetConfig): Configuration for the dataset.
        dataset_dir (str): Path to the directory containing the image files.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        label_mode=config.DATASET_LABEL_MODE,
        seed=config.DATASET_SEED,
        batch_size=config.DATASET_BATCH_SIZE,
        image_size=config.DATASET_IMAGE_SIZE,
        shuffle=shuffle,
    )
    dataset = preprocess_dataset(dataset)
    return dataset


def get_training_datasets(
    config: ImageDatasetConfig = ImageDatasetConfig(),
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates training and validation datasets based on the provided configuration.

    Args:
        config (ImageDatasetConfig, optional): Configuration for the datasets. Defaults to ImageDatasetConfig().

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    train_dataset = get_image_dataset(
        config=config,
        dataset_dir=config.DATASET_TRAIN_DIR,
        shuffle=True,
    )

    val_dataset = get_image_dataset(
        config=config,
        dataset_dir=config.DATASET_VAL_DIR,
        shuffle=False,
    )
    return train_dataset, val_dataset


def get_evaluation_dataset(
    config: ImageDatasetConfig = ImageDatasetConfig(),
) -> tuple[tf.data.Dataset, np.ndarray]:
    """
    Creates an evaluation dataset based on the provided configuration.

    Args:
        config (ImageDatasetConfig): Configuration for the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset object for evaluation.
        np.ndarray: The true labels for the evaluation dataset.
    """
    eval_dataset = get_image_dataset(
        config=config,
        dataset_dir=config.DATASET_EVALUATION_DIR,
        shuffle=False,
    )
    # return true labels for evaluation
    y_true = np.concatenate([y for x, y in eval_dataset], axis=0)
    return eval_dataset, np.argmax(y_true, axis=1)
