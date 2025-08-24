import os
import random


def get_random_image_path(class_folder: str, image_files: list) -> str:
    """Selects a random image path from a list of image files in a class folder.

    Args:
        class_folder (str): Path to the class folder.
        image_files (list): List of image file names in the class folder.

    Returns:
        str: Path to the randomly selected image file.
    """
    return os.path.join(class_folder, random.choice(image_files))
