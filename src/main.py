from time import time_ns

from keras.applications import densenet
from keras.applications import inception_resnet_v2
from keras.applications import mobilenet_v2
from keras.applications import nasnet
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception

from cnn.domain.models import ModelWrapper
from cnn.util.utils import *
from cnn.util.const import *

logging.basicConfig(level=LOG_LVL)
logger = logging.getLogger('cnn_timer')

TEST_IMG_WRAPPERS = load_img(TEST_IMG_FILENAMES)

MODELS = {
    VGG_16: ModelWrapper(vgg16.VGG16(), vgg16.preprocess_input, vgg16.decode_predictions, D_224),
    VGG_19: ModelWrapper(vgg19.VGG19(), vgg19.preprocess_input, vgg19.decode_predictions, D_224),
    MOBILENET: ModelWrapper(mobilenet_v2.MobileNetV2(), mobilenet_v2.preprocess_input, mobilenet_v2.decode_predictions,
                            D_224),
    INCEPTION: ModelWrapper(inception_resnet_v2.InceptionResNetV2(), inception_resnet_v2.preprocess_input,
                            inception_resnet_v2.decode_predictions, D_229),
    DENSENET: ModelWrapper(densenet.DenseNet(blocks=[6, 12, 48, 32]), densenet.preprocess_input,
                           densenet.decode_predictions,
                           D_224),
    XCEPTION: ModelWrapper(xception.Xception(), xception.preprocess_input, xception.decode_predictions, D_229),
    NASNET: ModelWrapper(nasnet.NASNet(), nasnet.preprocess_input, nasnet.decode_predictions, D_331),
}


def main():
    preprocess()
    for it in range(ITER):
        process_iter(it)
    write_csv()


def preprocess():
    for model_name, model_wrapper in MODELS.items():
        imgs = []
        for img in TEST_IMG_WRAPPERS:
            imgs.append(img.img_for_dim(model_wrapper.dim))
        model_wrapper.to_process = imgs


def process_iter(it):
    logger.info(f"ITER: {it + 1}")
    for model_name, model_wrapper in MODELS.items():
        start = time_ns()
        model_wrapper.process_mult()
        raw_out = model_wrapper.model.predict(model_wrapper.processed)
        predicted = model_wrapper.decode(raw_out)
        elapsed = time_ns() - start
        model_wrapper.results.time_results().add_time(elapsed)

        logger.info(f"Model: {model_name}, duration: {to_ms_unit(elapsed)}")
        for idx, it in enumerate(predicted):
            logger.info(f"{TEST_IMG_FILENAMES[idx]}:")
            for imagenet_id, label, likelihood in it:
                logger.info(f"Prediction: {label} - {likelihood:.2f}")
                model_wrapper.results.for_img(TEST_IMG_FILENAMES[idx]).add_prediction(label, likelihood)
        logger.info("")


def write_csv():
    out = f"Model Name{SEP}Average Time(ms)\n"
    for model_name, model_wrapper in MODELS.items():
        ti = model_wrapper.results.time_results().get_time_ms_avg()
        out += f"{model_name}{SEP}{ti}\n"
    out += "\n"

    for image_filename in TEST_IMG_FILENAMES:
        out += f"{image_filename}\n"

        out += f"\nModel Name{PRETTY_SEP}Label{SEP}Probability\n"

        for model_name, model_wrapper in MODELS.items():
            highest = model_wrapper.results.for_img(image_filename).get_label_highest()
            out += f"{model_name}{PRETTY_SEP}{highest[0]}{SEP}{highest[1]}\n"
        out += "\n"
    logger.info("")
    with open(OUTPUT_CSV, "w") as f:
        f.write(out)


main()
