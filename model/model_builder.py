from tensorflow import keras
from model.model import cls_model

def model_builder(backbone, image_size):
    input, output = cls_model(backbone='efficientV2-s',
                                     input_shape=(image_size[0], image_size[1], 3),
                                     classes=20)
    model = keras.Model(input, output)
    return model