import random
from typing import *

from torchvision.transforms import *
import numpy as np
import torchvision
from PIL import Image

from src.utils import get_logger

logger = get_logger("transforms")


def assert_percentage(percentage: float, name="percentage") -> None:
    assert 1.0 >= percentage >= 0.0, f"Invalid {name} value {percentage}. Must be in interval (0.0,1.0)"


def blur(spec: Dict, size: int, **kwargs) -> List:
    percentage = spec["percentage"]
    assert_percentage(percentage)
    # As per https://arxiv.org/abs/1711.08856
    blur_transform = [transforms.Resize(size // 4 + int((1.0 - percentage) * (size - size // 4)))]
    return blur_transform


def bernoulli_blur(spec: Dict, size: int, **kwargs) -> List:
    percentage = spec["percentage"]
    probability = spec["probability"]
    assert_percentage(percentage)
    assert_percentage(probability, "probability")
    rand_bool = random.uniform(0, 1) <= probability
    if rand_bool:
        # As per https://arxiv.org/abs/1711.08856
        blur_transform = [transforms.Resize(size // 4 + int((1.0 - percentage) * (size - size // 4)))]
    else:
        # As per https://arxiv.org/abs/1711.08856
        blur_transform = [transforms.Resize(size)]
    return blur_transform


def random_blur(spec: Dict, size: int, **kwargs) -> List:
    percentage = spec["percentage"]
    assert_percentage(percentage)
    percentage_random = random.uniform(percentage, 1)
    # Adapted from https://arxiv.org/abs/1711.08856
    blur_transform = [transforms.Resize(size // 4 + int((1.0 - percentage_random) * (size - size // 4)))]
    return blur_transform


def contrast(spec: Dict, **kwargs) -> List:
    strength: float = spec["strength"]
    assert_percentage(strength, "strength")
    contrast_transform = [transforms.ColorJitter(contrast=(strength * 0.9, strength * 0.9))]
    return contrast_transform


def random_contrast(spec: Dict, **kwargs) -> List:
    strength: float = spec["strength"]
    assert_percentage(strength, "strength")
    contrast_transform = [transforms.ColorJitter(contrast=strength* 0.90)]
    return contrast_transform


def random_crop(spec: Dict, size: int, **kwargs) -> List:
    crop_size: float = spec["crop_size"]
    assert 0 < crop_size < size, f"random crop transform: crop_size must be positive and smaller than original size."
    return [transforms.RandomCrop(size=crop_size, pad_if_needed=True)]


def random_resized_crop(spec: Dict, size, **kwargs) -> List:
    """
    A crop of random size (default: of 0.5 to 1.0) of the original size and a random aspect ratio
    (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size.
    :param spec:
    :param size:
    :return:
    """
    crop_percentage: float = spec.get("crop_percentage", 0.5)
    aspect_ratio_shift = spec.get("aspect_ratio_shift", 0.25)
    assert 0 < crop_percentage < 1
    assert 0 < aspect_ratio_shift < 1
    scale = (1 - crop_percentage, 1 + crop_percentage)
    ratio = (1 - aspect_ratio_shift, 1 + aspect_ratio_shift)
    return [transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)]


def random_horizontal_flip(spec: Dict, **kwargs) -> List:
    probability = spec["probability"]
    assert_percentage(probability, "probability")
    return [transforms.RandomHorizontalFlip(p=probability)]


def random_shift(spec: Dict, **kwargs) -> List:
    translate = spec["shift"]
    assert 0.0 <= translate <= 1.0, f"random shift transform: translate must be positive and <=1"
    return [transforms.RandomAffine(0, translate=(translate, translate))]


def random_rotate(spec: Dict, **kwargs) -> List:
    max_angle: Union[float, int] = spec["max_angle"]
    assert 180 > max_angle > -180, f"Invalid maximum angle {max_angle} for random rotate. " \
                                   f"Must be in interval (-180.0, 180.0)"

    return [transforms.RandomAffine(degrees=max_angle)]


def random_scale(spec: Dict, **kwargs) -> List:
    percentage = spec["percentage"]
    assert_percentage(percentage)
    return [transforms.RandomAffine(0, scale=(1 - percentage, 1 + percentage))]


def shear(spec: Dict, **kwargs) -> List:
    max_angle: Union[float, int] = spec["max_angle"]
    assert 90 >= max_angle >= -90, f"Invalid maximum angle {max_angle} for random rotate. " \
                                   f"Must be in interval (-90.0, 90.0)"

    return [transforms.RandomAffine(0, shear=max_angle)]


def random_grayscale(spec: Dict, **kwargs) -> List:
    probability = spec["probability"]
    assert_percentage(probability, "probability")
    return [transforms.RandomGrayscale(p=probability)]


def random_vertical_flip(spec: Dict, **kwargs) -> List:
    probability = spec["probability"]
    assert_percentage(probability, "probability")
    return [transforms.RandomVerticalFlip(p=probability)]


def random_rotation_90(spec: Dict, **kwargs) -> List:
    valid_angles: Tuple = (0, 90, 180, 270)
    angle: int = random.choice(valid_angles)
    return [transforms.RandomRotation(degrees=(angle, angle))]


def brightness(spec: Dict, **kwargs) -> List:
    # How much to adjust the brightness.
    # Can be any non negative number. 0 gives a black image, 1 gives the original image
    # while 2 increases the brightness by a factor of 2.
    brightness_strength: float = spec["strength"]
    assert 2.0 >= brightness_strength >= 0.1, f"brightness transform: invalid strength parameter {brightness_strength}, " \
                                              f"must be 2.0 >= strength >= 0.1"

    def adjust_brightness(img):
        return transforms.functional.adjust_brightness(img, brightness_strength)

    return [transforms.Lambda(adjust_brightness)]


def random_brightness(spec: Dict, **kwargs) -> List:
    # How much to jitter brightness.
    # brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
    # Should be non negative numbers.
    strength: float = spec["strength"]
    assert_percentage(strength, "strength")
    assert 0.9 >= strength >= 0.0, f"random brightness transform: invalid strength parameter {strength}, " \
                                   f"must be 0.9 >= strength >= 0.0"
    random_brightness_transform = [transforms.ColorJitter(brightness=strength)]
    return random_brightness_transform


def contrast(spec: Dict, **kwargs) -> List:
    # How much to adjust the contrast.
    # Can be any non negative number. 0 gives a solid gray image, 1 gives the original image
    # while 2 increases the contrast by a factor of 2.
    contrast_strength: float = spec["strength"]
    assert 2.0 >= contrast_strength >= 0.0, f"contrast transform: invalid strength parameter {contrast_strength}, " \
                                            f"must be 2.0 >= strength >= 0.0"

    def adjust_contrast(img):
        return transforms.functional.adjust_contrast(img, contrast_strength)

    return [transforms.Lambda(adjust_contrast)]


def random_contrast(spec: Dict, **kwargs) -> List:
    # How much to jitter contrast.
    # contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
    # Should be non negative numbers.
    strength: float = spec["strength"]
    assert_percentage(strength, "strength")
    assert 1.0 >= strength >= 0.0, f"random contrast transform: invalid strength parameter {strength}, " \
                                   f"must be 1.0 >= strength >= 0.0"
    random_contrast_transform = [transforms.ColorJitter(contrast=strength)]
    return random_contrast_transform


def saturation(spec: Dict, **kwargs) -> List:
    # How much to adjust the saturation.
    # 0 will give a black and white image, 1 will give the original image
    # while 2 will enhance the saturation by a factor of 2.
    strength: float = spec["strength"]
    assert 2.0 >= strength >= 0.1, f"saturation transform: invalid strength parameter {strength}, " \
                                   f"must be 2.0 >= strength >= 0.1"

    def adjust_saturation(img):
        return transforms.functional.adjust_saturation(img, strength)

    return [transforms.Lambda(adjust_saturation)]


def random_saturation(spec: Dict, **kwargs) -> List:
    # How much to jitter saturation.
    # saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
    # Should be non negative numbers.
    strength: float = spec["strength"]
    assert_percentage(strength, "strength")
    assert 0.9 >= strength >= 0.0, f"random saturation transform: invalid strength parameter {strength}, " \
                                   f"must be 0.9 >= strength >= 0.0"
    random_saturation_transform = [transforms.ColorJitter(saturation=strength)]
    return random_saturation_transform


def hue(spec: Dict, **kwargs) -> List:
    # How much to shift the hue channel.
    # Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in HSV space
    # in positive and negative direction respectively.
    # 0 means no shift.
    # Therefore, both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.
    hue_shift = spec["shift"]
    assert 0.5 >= hue_shift >= -0.5, f"hue transform: invalid shift parameter {hue_shift}, must be 0.5 >= shift >= 0.0"

    def adjust_hue(img):
        return transforms.functional.adjust_hue(img, hue_shift)

    return [transforms.Lambda(adjust_hue)]


def random_hue(spec: Dict, **kwargs) -> List:
    # How much to jitter hue.
    # hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
    # Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    max_shift = spec["max_shift"]
    assert 0.5 >= max_shift >= 0.0, f"hue transform: invalid max_shift parameter {max_shift}, " \
                                    f"must be 0.5 >= max_shift >= 0.0"

    return [transforms.ColorJitter(hue=max_shift)]


def random_gamma_correction(spec: Dict, **kwargs) -> List:
    max_shift = spec["max_shift"]
    assert 1 >= max_shift >= 0.0

    gamma = random.uniform(-1, 1) * max_shift
    gain = 1

    def adjust_gamma(img):
        return transforms.functional.adjust_gamma(img, gamma, gain)

    return [transforms.Lambda(adjust_gamma)]


TRANSFORM_GENERATORS = {
    "blur": blur,
    "random_blur": random_blur,
    "bernoulli_blur": bernoulli_blur,
    "random_crop": random_crop,
    "random_resized_crop": random_resized_crop,
    "random_shift": random_shift,
    "random_rotate": random_rotate,
    "shear": shear,
    "random_scale": random_scale,
    "random_grayscale": random_grayscale,
    "random_horizontal_flip": random_horizontal_flip,
    "random_vertical_flip": random_vertical_flip,
    "random_rotation_90": random_rotation_90,
    "brightness": brightness,
    "random_brightness": random_brightness,
    "contrast": contrast,
    "random_contrast": random_contrast,
    "saturation": saturation,
    "random_saturation": random_saturation,
    "hue": hue,
    "random_hue": random_hue,
    "random_gamma_correction": random_gamma_correction,
}


def assemble_transforms(purpose: str, transform_specs: Dict, size_spec, to_pil: bool = False):
    transform_sequence: List = []

    if to_pil:
        transform_sequence.append(transforms.ToPILImage())

    for transform_name in transform_specs:
        try:
            transform_generator = TRANSFORM_GENERATORS[transform_name]
        except KeyError:
            raise ValueError(f"Invalid transform name {transform_name}. "
                             f"Must be one of {list(TRANSFORM_GENERATORS.keys())}")
        try:
            transform_sequence.extend(transform_generator(transform_specs[transform_name],
                                                          size=size_spec))
        except KeyError:
            raise KeyError(f"Incomplete transform spec for {transform_name}: {transform_specs[transform_name]}")

    resize_transform = transforms.Resize(size_spec)
    transform_sequence.append(resize_transform)

    transform_sequence.append(transforms.ToTensor())

    logger.info(f"Assembled transforms for purpose {purpose}: {transform_sequence}")
    transformation = transforms.Compose(transform_sequence)

    return transformation
