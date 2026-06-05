from typing import Any, Callable

class DictImageTransform:
    """
    A class to transform the image in a dictionary.

    This is needed for the ssl4eo dataset since get_item returns a dictionary.

    Args:
        image_transform (Callable): The transformation to apply to the image.

    Returns:
        dict: The transformed sample.
    """

    def __init__(self, image_transform: Callable[[Any], Any]) -> None:
        self.image_transform = image_transform

    def __call__(self, sample: dict) -> dict:
        # Only transform the 'image' key, leave others unchanged
        sample = sample.copy()
        sample["image"] = self.image_transform(sample["image"])
        return sample

