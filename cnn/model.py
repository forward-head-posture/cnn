# pylint: disable=import-error

from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def get_model(base_model_name="InceptionResNetV2"):
    if base_model_name == "MobileNetV2":
        base_model = MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
    else:
        base_model = MobileNet(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(1, activation="linear")(x)
    return Model(inputs=base_model.input, outputs=outputs)
