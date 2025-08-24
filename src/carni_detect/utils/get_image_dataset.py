import tensorflow as tf


def preprocess_dataset(dataset: tf.data.Dataset):
    dataset = dataset.map(
        lambda image, label: (
            tf.keras.applications.resnet50.preprocess_input(image),
            label,
        )
    )
    return dataset


def get_image_dataset(
    data_directory: str, batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from the images in the specified directory.

    Args:
        data_directory (str): Path to the directory containing the image files.
        batch_size (int): Number of images to include in each batch.

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        label_mode="categorical",
        seed=42,
        batch_size=batch_size,
        image_size=(224, 224),  # Resize images to a consistent size
        shuffle=shuffle,
        color_mode="rgb",
    )
    dataset = preprocess_dataset(dataset)
    return dataset
