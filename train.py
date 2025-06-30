import tensorflow as tf
from models.resnet import res_net
from data.dataset import load_cifar
from utils.callbacks import cosine_decay_scheduler, get_tensorboard_callback

(x_train, y_train), (x_test, y_test), augment = load_cifar()

model = res_net()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    cosine_decay_scheduler(100, 0.1),
    get_tensorboard_callback()
]

model.fit(
    augment(x_train), y_train,
    validation_split = 0.1,
    epoch = 100,
    batch_size = 128,
    callbacks = callbacks
)

model.save('resnet18_cifar10.h5')