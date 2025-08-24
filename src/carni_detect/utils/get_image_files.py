import os


def get_image_files(class_folder: str, image_format: tuple) -> list:
    """Retrieves a list of image files in a given class folder.

    Args:
        class_folder (str): Path to the class folder.

    Returns:
        list: List of image file names in the class folder.
    """
    return [file for file in os.listdir(class_folder) if file.endswith((image_format))]
