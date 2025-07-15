# [keras] 데이터_증강_Dogs_vs_Cats



# Dogs vs Cats 데이터셋
# pip install gdown



import gdown
import zipfile
import os

if not os.path.isdir('cats_vs_dogs_small'):
    gdown.download(id='1z2WPTBUI-_Q2jZtcRtQL0Vxigh-z6dyW', output='cats_vs_dogs_small.zip')
    cats_vs_dogs_small = zipfile.ZipFile('cats_vs_dogs_small.zip')
    cats_vs_dogs_small.extractall()
    cats_vs_dogs_small.close()



print(sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/train/cat")))



print("train : "+sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/train/cat"))[0]+" ~ "+sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/train/cat"))[-1])
print("validation : "+sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/validation/cat"))[0]+" ~ "+sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/validation/cat"))[-1])
print("test : "+sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/test/cat"))[0]+" ~ "+sorted(os.listdir("./transfer_learning/cats_vs_dogs_small/test/cat"))[-1])



import matplotlib.pyplot as plt
import matplotlib.image as image

plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    img_path = f'./transfer_learning/cats_vs_dogs_small/train/cat/cat.{i}.jpg'
    img = image.imread(img_path)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.show()



plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    img_path = f'./transfer_learning/cats_vs_dogs_small/train/dog/dog.{i}.jpg'
    img = image.imread(img_path)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.show()



# 입력 파이프라인 API
x = ["a","b","c"]
iterator = iter(x)
print(next(iterator))
print(next(iterator))
print(next(iterator))



import numpy as np
import tensorflow as tf

dataset = np.arange(100).reshape(20,5)
print(dataset)

dataset = tf.data.Dataset.from_tensor_slices(dataset)

for data in dataset:
    print(data)



batched_dataset = dataset.batch(4)

for batch in batched_dataset:
    print(batch)



import pathlib
from tensorflow.keras.utils import image_dataset_from_directory

base_dir = pathlib.Path("./transfer_learning/cats_vs_dogs_small")

train_dataset = image_dataset_from_directory(
    base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    base_dir / "test",
    image_size=(180, 180),
    batch_size=32)



iterator = iter(train_dataset)
batch_1 = next(iterator)
print(batch_1)



for data in train_dataset:
    print(data[1])



plt.figure(figsize=(15,30))
for i in range(32):
    plt.subplot(8,4,i+1)
    plt.imshow(batch_1[0][i]/255)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(batch_1[1][i].numpy())
plt.show()



# 적은 데이터로 학습
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Rescaling, Flatten, Dense

inputs = keras.Input(shape=(180, 180, 3))
x = Rescaling(1./255)(inputs)
x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = Flatten()(x)
outputs = Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()



model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = keras.callbacks.ModelCheckpoint(
    filepath="convnet_from_scratch.keras",
    save_best_only=True,
    monitor="val_loss")

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)



import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()



test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"테스트 정확도: {test_acc:.3f}")



# 데이터 증강
from keras.layers import RandomFlip, RandomRotation, RandomZoom

data_augmentation = keras.Sequential(
    [RandomFlip("horizontal"),
     RandomRotation(0.1),
     RandomZoom(0.2)])



img = batch_1[0][0]

plt.imshow(img/255)
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(figsize=(12, 12))
for i in range(25):
    augmented_img = data_augmentation(img[np.newaxis,:,:,:])
    plt.subplot(5, 5, i + 1)
    plt.imshow(augmented_img[0]/255)
    plt.xticks([])
    plt.yticks([])
plt.show()



from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
img = train_images[0]

mnist_rotation = keras.Sequential(
    [RandomRotation(0.1, fill_mode='nearest')])

plt.imshow(img/255, cmap='binary')
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(figsize=(12, 12))
for i in range(25):
    augmented_img = mnist_rotation(img[np.newaxis,:,:,np.newaxis])
    plt.subplot(5, 5, i + 1)
    plt.imshow(augmented_img[0,:,:,:]/255, cmap='binary')
    plt.xticks([])
    plt.yticks([])
plt.show()



from keras.layers import RandomTranslation

mnist_translation = keras.Sequential(
    [RandomTranslation(0.2,0.2,fill_mode='nearest')])

plt.imshow(img/255, cmap='binary')
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(figsize=(12, 12))
for i in range(25):
    augmented_img = mnist_translation(img[np.newaxis,:,:,np.newaxis])
    plt.subplot(5, 5, i + 1)
    plt.imshow(augmented_img[0,:,:,:]/255, cmap='binary')
    plt.xticks([])
    plt.yticks([])
plt.show()



# 데이터 증강을 통한 학습
from keras.layers import Dropout

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = Rescaling(1./255)(x)
x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()



# 7~8 분 정도 소요됩니다.
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint(
    filepath="convnet_from_scratch_with_augmentation.keras",
    save_best_only=True,
    monitor="val_loss")]

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)



# 데이터 증강을 통해 대략 70% → 80%의 성능 향상
test_model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"테스트 정확도: {test_acc:.3f}")