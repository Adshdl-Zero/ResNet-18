import tensorflow as tf
from data.dataset import load_cifar

model = tf.keras.models.load_model('resnet18_cifar10.h5')

(_, _), (x_test, y_test), _ = load_cifar()

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
