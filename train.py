import tensorflow as tf
from sklearn.model_selection import train_test_split
from models.resnet import res_net
from data.dataset import load_cifar
from utils.callbacks import cosine_decay_scheduler, get_tensorboard_callback

(x_train, y_train), (x_test, y_test), augment = load_cifar()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(64).map(lambda x, y: (augment(x, training=True), y))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64).prefetch(tf.data.AUTOTUNE)

model = res_net()
lr = 0.1
no_of_epochs = 120
mom = 0.9
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=True),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

callbacks = [
    cosine_decay_scheduler(no_of_epochs, lr),
    get_tensorboard_callback()
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=no_of_epochs,
    callbacks=callbacks
)

model.save('resnet18_cifar10.keras')