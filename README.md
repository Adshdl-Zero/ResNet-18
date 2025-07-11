# 🧠 ResNet-18 CIFAR-10 Image Classifier

Trained a ResNet-18-like convolutional neural network on the CIFAR-10 dataset to classify 32x32 images into 10 object categories with high accuracy.

**📊 Test Accuracy**: **90.44%**
**📘 Original Paper Accuracy**: **~91.25%**

---

## ⚙️ How It's Made:

**Tech Used**: Python, TensorFlow, Keras, Scikit-learn

* Built a ResNet-18 architecture from scratch using TensorFlow and Keras Functional API
* Used residual blocks with skip connections to prevent vanishing gradients in deep networks
* Applied data augmentation (random flip, crop, normalization) to increase generalization
* Trained using `SparseCategoricalCrossentropy` for multi-class classification

---

## 🚀 Optimizations:

* **L2 Regularization** added to Conv2D layers to reduce overfitting
* **Batch Normalization** for faster convergence and better performance
* **Cosine Decay Learning Rate Scheduler** implemented for efficient training
* Performed hyperparameter tuning (batch size, learning rate, augmentation techniques)

---

## 📚 Lessons Learned:

* Learned the structure and advantages of **Residual Blocks**
* Understood how **deep networks can be trained more effectively** using skip connections
* Gained experience with **modular ML project design** and clean code organization
* Explored effects of **different optimizers, schedulers, and regularizers** on model accuracy
* Practiced using `.keras` format to save/load Keras models

---

## 📦 Results & Code Structure

* ✅ Achieved **90.44%** accuracy on test set with trained ResNet model
* ⛏️ Evaluated saved model using `validate.py`
* 📂 Modular file structure:

  * `model.py` – defines ResNet-18 model with residual blocks
  * `train.py` – handles model training and saving
  * `validate.py` – evaluates test accuracy using saved model
  * `dataset.py` – loads and augments CIFAR-10 data

---
