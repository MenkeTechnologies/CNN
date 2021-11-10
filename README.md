# CNN Timer

## Times various CNN models with same images and same imagenet training set

# Models Tested

```
MODELS = {
    VGG_16: ModelWrapper(vgg16.VGG16(), vgg16.preprocess_input, vgg16.decode_predictions, D_224),
    VGG_19: ModelWrapper(vgg19.VGG19(), vgg19.preprocess_input, vgg19.decode_predictions, D_224),
    MOBILENET: ModelWrapper(mobilenet_v2.MobileNetV2(), mobilenet_v2.preprocess_input, mobilenet_v2.decode_predictions, D_224),
    INCEPTION: ModelWrapper(inception_resnet_v2.InceptionResNetV2(), inception_resnet_v2.preprocess_input,
                            inception_resnet_v2.decode_predictions, D_229),
    DENSENET: ModelWrapper(densenet.DenseNet(blocks=[6, 12, 48, 32]), densenet.preprocess_input, densenet.decode_predictions,
                           D_224),
    XCEPTION: ModelWrapper(xception.Xception(), xception.preprocess_input, xception.decode_predictions, D_229),
    NASNET: ModelWrapper(nasnet.NASNet(), nasnet.preprocess_input, nasnet.decode_predictions, D_331),
}
```

# To generate output CSV for graphing

```
python3 -m pip install -r requirements.txt
python3 src/main.py
```

CSV output file = ENC3246CNNComparison.csv

# created by MenkeTechnologies
