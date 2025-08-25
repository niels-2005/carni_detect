from fastapi import UploadFile
import tensorflow as tf
import io


async def preprocess_image(file: UploadFile, image_size: tuple[int, int]) -> tf.Tensor:
    """
    Preprocesses an uploaded image file for input into a ResNet50 model.

    Reads the contents of an uploaded image file asynchronously, resizes it to the specified dimensions,
    converts it to a NumPy array, expands its dimensions to match the expected input shape,
    and applies ResNet50-specific preprocessing.

    Args:
        file (UploadFile): The uploaded image file to preprocess.

    Returns:
        tf.Tensor: The preprocessed image tensor ready for model inference.
    """
    contents = await file.read()
    image = tf.keras.preprocessing.image.load_img(
        io.BytesIO(contents),
        target_size=image_size,
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image
