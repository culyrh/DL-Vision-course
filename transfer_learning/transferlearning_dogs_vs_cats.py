# [keras] TransferLearning_Dogs_vs_Cats



# 사전 훈련된 모델
from tensorflow import keras

dir(keras.applications)



vgg16 = keras.applications.vgg16.VGG16()
vgg16.summary()



from tensorflow.keras.utils import plot_model

plot_model(vgg16)



import gdown, zipfile, os

if not os.path.isfile('VGG16_test.zip'):
    gdown.download(id='11LZAFSFVtDsdKdLcFR9E-MaDoar6C3R5', output='VGG16_test.zip')
    VGG16_test = zipfile.ZipFile('VGG16_test.zip')
    VGG16_test.extractall()
    VGG16_test.close()



with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

print(labels)



import matplotlib.pyplot as plt
import matplotlib.image as image

img = image.imread("golden.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)



import tensorflow as tf
import numpy as np

img = tf.image.resize(img,(224,224))
out=vgg16(tf.keras.applications.vgg16.preprocess_input(tf.expand_dims(img,axis=0)))
print(out)



top_5 = np.argsort(-out[0])[:5]
for idx in top_5:
    print(f"{labels[idx]} : {out[0,idx]*100:.3f}%")



from sklearn.datasets import load_sample_image

img = load_sample_image('china.jpg')
print(img.shape)

plt.imshow(img)
plt.axis('off')
plt.show()



img = tf.image.resize(img,(224,224))
out=vgg16(tf.keras.applications.vgg16.preprocess_input(img[np.newaxis,:,:,:]))

top_5 = np.argsort(-out[0])[:5]
for idx in top_5:
    print(f"{labels[idx]} : {out[0,idx]*100:.4f}%")



img = load_sample_image('flower.jpg')
print(img.shape)

plt.imshow(img)
plt.axis('off')
plt.show()



img = tf.image.resize(img,(224,224))
out=vgg16(tf.keras.applications.vgg16.preprocess_input(img[np.newaxis,:,:,:]))

top_5 = np.argsort(-out[0])[:5]
for idx in top_5:
    print(f"{labels[idx]} : {out[0,idx]*100:.4f}%")



img = image.imread("./golden.jpg")
img = tf.image.resize(img,(224,224))
plt.imshow(img)
out=vgg16(tf.keras.applications.vgg16.preprocess_input(tf.expand_dims(img,axis=0)))

top_20 = np.argsort(-out[0])[:20]
for idx in top_20:
    print(f"{labels[idx]} : {out[0,idx]*100:.3f}%")



# 실습1 (i) labrador.jpg를 출력하시오.
# VGG16이 예측하는 톱 5의 클래스 이름과 확률을 출력하시오.
img = image.imread("./labrador.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()

img = tf.image.resize(img,(224,224))
out=vgg16(tf.keras.applications.vgg16.preprocess_input(img[np.newaxis,:,:,:]))

top_5 = np.argsort(-out[0])[:5]
for idx in top_5:
    print(f"{labels[idx]} : {out[0,idx]*100:.4f}%")



# (ii) strawberries.jpg를 출력하시오. 
# VGG16이 예측하는 톱 5의 클래스 이름과 확률을 출력하시오.
img = image.imread("./strawberries.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()

img = tf.image.resize(img,(224,224))
out=vgg16(tf.keras.applications.vgg16.preprocess_input(img[np.newaxis,:,:,:]))

top_5 = np.argsort(-out[0])[:5]
for idx in top_5:
    print(f"{labels[idx]} : {out[0,idx]*100:.4f}%")



# (iii) car.jpg를 출력하시오. 
# VGG16이 예측하는 톱 5의 클래스 이름과 확률을 출력하시오.
img = image.imread("./car.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()

img = tf.image.resize(img,(224,224))
out=vgg16(tf.keras.applications.vgg16.preprocess_input(img[np.newaxis,:,:,:]))

top_5 = np.argsort(-out[0])[:5]
for idx in top_5:
    print(f"{labels[idx]} : {out[0,idx]*100:.4f}%")



# 전이 학습
conv_base = keras.applications.vgg16.VGG16(
    include_top=False,
    input_shape=(180, 180, 3))

conv_base.trainable = False

conv_base.summary()



import pathlib
from tensorflow.keras.utils import image_dataset_from_directory

if not os.path.isdir('cats_vs_dogs_small'):
    gdown.download(id='1z2WPTBUI-_Q2jZtcRtQL0Vxigh-z6dyW', output='cats_vs_dogs_small.zip')
    cats_vs_dogs_small = zipfile.ZipFile('cats_vs_dogs_small.zip')
    cats_vs_dogs_small.extractall()
    cats_vs_dogs_small.close()

base_dir = pathlib.Path("cats_vs_dogs_small")

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



from keras.layers import Flatten, Dense, Dropout

inputs = keras.Input(shape=(180, 180, 3))
x = keras.applications.vgg16.preprocess_input(inputs)
x = conv_base(x)
x = Flatten()(x)
x = Dense(256)(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)



model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics="accuracy")

callbacks = [keras.callbacks.ModelCheckpoint(
      filepath="feature_extraction.keras",
      save_best_only=True,
      monitor="val_loss")]

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)



import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()



test_model = keras.models.load_model(
    "feature_extraction.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"테스트 정확도: {test_acc:.3f}")



# 실습2 특성추출기의 파라미터를 동결시키지 않고 10에폭동안 훈련시키시오.
conv_base = keras.applications.vgg16.VGG16(
    include_top=False,
    input_shape=(180, 180, 3))

inputs = keras.Input(shape=(180, 180, 3))
x = keras.applications.vgg16.preprocess_input(inputs)
x = conv_base(x)
x = Flatten()(x)
x = Dense(256)(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

model.fit(train_dataset,
          epochs=10,
          validation_data=validation_dataset)



# 증강 데이터 + 전이 학습
from keras import layers

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"),
     layers.RandomRotation(0.1),
     layers.RandomZoom(0.2)])

conv_base = keras.applications.vgg16.VGG16(
    include_top=False,
    input_shape=(180, 180, 3))
conv_base.trainable = False

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = Flatten()(x)
x = Dense(256)(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)



model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_data_augmentation.h5",
        save_best_only=True,
        monitor="val_loss")]

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)



acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()



test_model = keras.models.load_model(
    "feature_extraction_with_data_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"테스트 정확도: {test_acc:.3f}")



# 미세 조정(fine tuning)
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False

print("동결 해제층 :")
for layer in conv_base.layers:
    if layer.trainable:
        print(layer.name)

print("\n")
conv_base.summary()



model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint(
        filepath="fine_tuning.keras",
        save_best_only=True,
        monitor="val_loss")]

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)



model = keras.models.load_model("fine_tuning.keras")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"테스트 정확도: {test_acc:.3f}")



errors_cat = []
for i in range(1000):
    image_path = f'./cats_vs_dogs_small/test/cat/cat.{1500+i}.jpg'
    img = plt.imread(image_path)
    img = tf.image.resize(img,(180,180))
    img = tf.expand_dims(img,axis=0)
    if model(img)[0]>0.5:
        errors_cat.append(i)

print(errors_cat)



import math

plt.figure(figsize=(16, 4*math.ceil(len(errors_cat)/4)))
for i in range(len(errors_cat)):
    plt.subplot(math.ceil(len(errors_cat)/4),4,i+1)
    image_path = f'./cats_vs_dogs_small/test/cat/cat.{1500+errors_cat[i]}.jpg'
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()



# 실습3 (i) 신경망이 틀리게 대답한 강아지 이미지의 인덱스와 개수를 출력하시오.
errors_dog = []
for i in range(1000):
    image_path = f'./cats_vs_dogs_small/test/dog/dog.{1500+i}.jpg'
    img = plt.imread(image_path)
    img = tf.image.resize(img,(180,180))
    img = tf.expand_dims(img,axis=0)
    if model(img)[0]<0.5:
        errors_dog.append(i)

print(errors_dog)
print(len(errors_dog))



# (ii) 신경망이 틀리게 대답한 강아지 이미지들을 모아찍기로 출력하시오.
plt.figure(figsize=(16, 4*math.ceil(len(errors_dog)/4)))
for i in range(len(errors_dog)):
    plt.subplot(math.ceil(len(errors_dog)/4),4,i+1)
    image_path = f'./cats_vs_dogs_small/test/dog/dog.{1500+errors_dog[i]}.jpg'
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()