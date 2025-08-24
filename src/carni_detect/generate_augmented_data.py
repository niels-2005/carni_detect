import cv2
import os
from src.carni_detect.utils.augment_images import augment_image
from src.carni_detect.utils.get_random_image import (
    get_random_image_path,
)
from src.carni_detect.utils.get_image_files import get_image_files


class DataAugmentationConfig:
    """Configuration class for data augmentation settings.

    Attributes:
        SAMPLED_DATA_PATH (str): Path to the sampled data directory.
        DATASET_PATH (str): Path to the dataset directory.
        DATASET_TRAIN_PATH (str): Path to the training dataset directory.
        MAX_IMAGES (int): Maximum number of images per class after augmentation.
        IMAGE_FORMAT (str): Format of the image files to process.
    """

    SAMPLED_DATA_PATH = "sampled_data"
    DATASET_PATH = "dataset"
    DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    MAX_IMAGES_PER_CLASS = 1000
    IMAGE_FORMAT = (".png", ".jpg", ".jpeg")


def generate_augmented_data(config: DataAugmentationConfig) -> None:
    """Generates augmented images for each class in the training dataset.

    For each class folder in the training dataset, this function generates
    augmented images using the `augment_image` function until the number of
    images in the folder reaches `Config.MAX_IMAGES`.

    The augmented images are saved in the same class folder with unique names.
    """
    for class_name in os.listdir(config.DATASET_TRAIN_PATH):
        class_folder = os.path.join(config.DATASET_TRAIN_PATH, class_name)
        image_files = get_image_files(class_folder, config.IMAGE_FORMAT)

        # Generate Augmented Images till Config.MAX_IMAGES_PER_CLASS
        while len(image_files) < config.MAX_IMAGES_PER_CLASS:
            random_image_path = get_random_image_path(class_folder, image_files)
            image = cv2.imread(random_image_path)
            augmented_image = augment_image(image)

            # Save the augmented image
            new_image_name = f"aug_{len(image_files)}_{random_image_path}"
            new_image_path = os.path.join(class_folder, new_image_name)
            cv2.imwrite(new_image_path, augmented_image)

            image_files.append(new_image_name)


if __name__ == "__main__":
    config = DataAugmentationConfig()
    generate_augmented_data(config)
