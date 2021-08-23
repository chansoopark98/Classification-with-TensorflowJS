import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import argparse
import time
from utils.load_datasets import GenerateDatasets
from utils.callbacks import Scalar_LR
from model.model_builder import model_builder
from model.loss import Loss

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=120)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0001)
parser.add_argument("--image_size",   type=int, help="입력 이미지 크기 설정", default=224)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정 mirror or multi", default='mirror')

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
IMAGE_SIZE = (args.image_size, args.image_size)
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir

USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

train_dataset_config = GenerateDatasets(mode='train', data_dir=DATASET_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
valid_dataset_config = GenerateDatasets(mode='valid', data_dir=DATASET_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)
valid_data = valid_dataset_config.get_validData(valid_dataset_config.valid_data)

steps_per_epoch = train_dataset_config.number_train // BATCH_SIZE
validation_steps = valid_dataset_config.number_valid // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)


checkpoint_val_loss = ModelCheckpoint(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_best_loss.h5',
                                      monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)

polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=0.00005, power=0.9)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay, verbose=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

if MIXED_PRECISION:
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

callback = [checkpoint_val_loss, tensorboard, testCallBack, lr_scheduler]

loss = Loss(batch_size=BATCH_SIZE)

model = model_builder(backbone=None, image_size=(IMAGE_SIZE[0],IMAGE_SIZE[1], 3))

if USE_WEIGHT_DECAY:
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY / 2)
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, regularizer)

model.compile(
    optimizer=optimizer,
    metrics='accuracy',
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    loss=loss.sparse_categorical_loss,)

model.summary()

history = model.fit(train_data,
                    validation_data=valid_data,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=callback)





