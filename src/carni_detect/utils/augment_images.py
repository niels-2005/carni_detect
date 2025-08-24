import random
import cv2
import numpy as np


def randomly_flip_image(image):
    """Randomly flips an image horizontally with a 50% probability.

    Args:
        image (numpy.ndarray): Input image to be flipped.

    Returns:
        numpy.ndarray: The flipped image or the original image if no flip is applied.
    """
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    return image


def randomly_rotate_image(image):
    """Randomly rotates an image by a random angle between -15 and 15 degrees.

    Args:
        image (numpy.ndarray): Input image to be rotated.

    Returns:
        numpy.ndarray: The rotated image.
    """
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return image


def randomly_adjust_brightness(image):
    """Randomly adjusts the brightness of an image with a 50% probability.

    Args:
        image (numpy.ndarray): Input image to be adjusted.

    Returns:
        numpy.ndarray: The brightness-adjusted image or the original image if no adjustment is applied.
    """
    if random.random() > 0.5:
        factor = 1.0 + (random.random() - 0.5) * 0.2
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image


def randomly_scale_image(image):
    """Randomly scales an image by a factor between 0.8 and 1.2 with a 50% probability.

    Args:
        image (numpy.ndarray): Input image to be scaled.

    Returns:
        numpy.ndarray: The scaled image resized back to the original dimensions.
    """
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        image = cv2.resize(image, (w, h))  # Resize back to original size
    return image


def randomly_crop_image(image):
    """Randomly crops and resizes an image with a 50% probability.

    Args:
        image (numpy.ndarray): Input image to be randomly cropped.

    Returns:
        numpy.ndarray: The cropped (and resized) image, or the original image if no cropping is performed.
    """
    if random.random() > 0.5:
        h, w = image.shape[:2]
        crop_size = random.uniform(0.8, 1.0)
        crop_h, crop_w = int(h * crop_size), int(w * crop_size)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        image = image[start_h : start_h + crop_h, start_w : start_w + crop_w]
        image = cv2.resize(image, (w, h))  # Resize back to original size
    return image


def randomly_color_jitter(image):
    """Randomly applies color jitter to an image with a 50% probability.

    Args:
        image (numpy.ndarray): Input image to be color jittered.

    Returns:
        numpy.ndarray: The color-jittered image or the original image if no jitter is applied.
    """
    if random.random() > 0.5:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = np.array(hsv_image, dtype=np.float64)
        hsv_image[:, :, 1] *= random.uniform(0.8, 1.2)  # Saturation
        hsv_image[:, :, 2] *= random.uniform(0.8, 1.2)  # Brightness
        hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image


def augment_image(image):
    """Applies a series of random augmentations to an image.

    The augmentations include horizontal flip, rotation, brightness adjustment,
    scaling, cropping, and color jitter.

    Args:
        image (numpy.ndarray): Input image to be augmented.

    Returns:
        numpy.ndarray: The augmented image.
    """
    image = randomly_flip_image(image)
    image = randomly_rotate_image(image)
    image = randomly_adjust_brightness(image)
    image = randomly_scale_image(image)
    image = randomly_crop_image(image)
    image = randomly_color_jitter(image)
    return image
