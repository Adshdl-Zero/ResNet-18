# %tensorflow_version 2.x
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard # type: ignore
import math

def get_tensorboard_callback(log_dir='logs'):
    return TensorBoard(log_dir=log_dir)

def cosine_decay_scheduler(max_epochs, initial_lr):
    def scheduler(epoch):
        return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    return LearningRateScheduler(scheduler)
