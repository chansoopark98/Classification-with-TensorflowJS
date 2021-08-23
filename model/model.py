from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Softmax, Conv2D, Dropout

def cls_model(backbone='efficientV2-s', input_shape=(224, 224, 3), classes=20, OS=16):
    base = MobileNetV2(input_shape=input_shape, include_top=True, weights='imagenet', classifier_activation=None)
    top = base.get_layer('predictions').output
    top = Dropout(0.5)(top)
    #
    # top = GlobalAveragePooling2D()(top)
    top = Dense(classes)(top)
    base.summary()
    return base.input, top

