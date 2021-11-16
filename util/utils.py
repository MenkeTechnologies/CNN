import os

from keras.preprocessing import image

from cnn_comparison.domain import models
from cnn_comparison.util.const import *


def to_ms_unit(raw):
    return f"{to_ms(raw)}ms"


def to_ms(raw):
    return f"{raw * NS_TO_MS:.2f}"


def to_s(raw):
    return f"{raw * NS_TO_S:.2f}s"


def load_img(images, MODULE_ROOT_DIR):
    li = []
    for filename in images:
        abs_path = os.path.join(MODULE_ROOT_DIR, IMAGE_DIR, filename)

        li.append(
            models.ImageFileToNumPyAry(
                filename,
                image.img_to_array(image.load_img(abs_path, target_size=(224, 224))),
                image.img_to_array(image.load_img(abs_path, target_size=(299, 299))),
                image.img_to_array(image.load_img(abs_path, target_size=(331, 331)))
            )
        )
    return li
