import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as image
import numpy as np

img_path = keras.utils.get_file(
    fname="cat.jpg",
    origin="https://img-datasets.s3.amazonaws.com/cat.jpg")
img = image.imread(img_path)

plt.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)



#augmentation이 적용된 image들을 시각화 해주는 함수
def show_aug_image_batch(image, generator, n_images=4):

    # ImageDataGenerator는 여러개의 image를 입력으로 받기 때문에 4차원으로 입력 해야함.
    image_batch = np.expand_dims(image, axis=0)

    # featurewise_center or featurewise_std_normalization or zca_whitening 가 True일때만 fit 해주어야함
    # generator.fit(image_batch)
    # flow로 image batch를 generator에 넣어주어야함.
    data_gen_iter = generator.flow(image_batch)

    fig, axs = plt.subplots(nrows=1, ncols=n_images, figsize=(24, 8))

    for i in range(n_images):
    	#generator에 batch size 만큼 augmentation 적용(매번 적용이 다름)
        aug_image_batch = next(data_gen_iter)
        aug_image = np.squeeze(aug_image_batch)
        aug_image = aug_image.astype('int')
        axs[i].imshow(aug_image)
        axs[i].axis('off')
    
    plt.show()



data_generator = ImageDataGenerator(horizontal_flip=True)
show_aug_image_batch(img, data_generator, n_images=4)



data_generator = ImageDataGenerator(vertical_flip=True)
show_aug_image_batch(img, data_generator, n_images=4)



data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
show_aug_image_batch(img, data_generator, n_images=4)



data_generator = ImageDataGenerator(rotation_range=45)
show_aug_image_batch(img, data_generator, n_images=4)