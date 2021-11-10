import logging

TRAFFIC_LIGHT_PNG = "traffic_light.png"
STOP_PNG = "stop_sign.png"
SEDAN_PNG = "sedan.png"

TEST_IMG_FILENAMES = [
    TRAFFIC_LIGHT_PNG,
    STOP_PNG,
    SEDAN_PNG
]

LOG_LVL = logging.INFO
ITER = 5

NS_TO_MS = 1 / 1e6
NS_TO_S = 1 / 1e9

PRETTY_SEP = "/"
SEP = ","

D_224 = 224
D_229 = 299
D_331 = 331
VGG_19 = "VGG19"
VGG_16 = "VGG16"
MOBILENET = "MobileNetV2"
INCEPTION = "InceptionResNetV2"
DENSENET = "DenseNet"
XCEPTION = "Xception"
NASNET = "NASNet"
IMAGE_DIR = "test_img"
OUTPUT_CSV = "data.csv"
