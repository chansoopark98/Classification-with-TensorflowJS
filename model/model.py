from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Softmax, Conv2D

def cls_model(backbone='efficientV2-s', input_shape=(224, 224, 3), classes=20, OS=16):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    top = base.get_layer('block_16_project_BN').output

    top = GlobalAveragePooling2D()(top)
    top = Dense(classes)(top)
    base.summary()
    return base.input, top

