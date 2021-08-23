import tensorflow as tf
import psutil

class Scalar_LR(tf.keras.callbacks.Callback):
    def __init__(self, name, TENSORBOARD_DIR):
        super().__init__()
        self.name = name
        self.file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
        self.file_writer.set_as_default()


    def on_epoch_end(self, epoch, logs=None):
        logs['learning rate'] = self.model.optimizer.lr
        tf.summary.scalar("learning rate", logs['learning rate'], step=epoch)
        print(psutil.virtual_memory().used / 2 ** 30)