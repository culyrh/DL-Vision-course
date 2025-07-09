# 1. CNN_from_scratch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import urllib
import requests
from PIL import Image
from io import BytesIO
# import sys
# from matplotlib.image import imread
# img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'
# img = url_to_image(img_url)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()




# util functions
def url_to_image(url, gray=False):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    if gray:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def filtered_image(image, filter, output_size):
    filtered_image = np.zeros((output_size, output_size))
    filter_size = filter.shape[0]

    for i in range(output_size):
        for j in range(output_size):
            multiply_values = image[i:(i+filter_size), j:(j+filter_size)] * filter
            sum_value = np.sum(multiply_values)

            if (sum_value > 255):   # 최대치 255
                sum_value = 255
            filtered_image[i, j] = sum_value

    return filtered_image



# 이미지 확인
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img = url_to_image(img_url, gray=True)
print('img shape:', img.shape)
plt.imshow(img, cmap='gray')
plt.show()



# 필터연산 적용
horizontal_filter  = np.array([[1., 2., 1.],
                            [0., 0., 0.],
                            [-1., -2., -1.]])

vertical_filter= np.array([[1., 0., -1.],
                              [2., 0., -2.],
                              [1., 0., -1.]])

output_size = int((img.shape[0] - 3) / 1+1)
print('output size:', output_size)

vertical_filtered = filtered_image(img, vertical_filter, output_size)
horizontal_filtered = filtered_image(img, horizontal_filter, output_size)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.title('Vertical')
plt.imshow(vertical_filtered, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Horizontal')
plt.imshow(horizontal_filtered, cmap='gray')
plt.show()



# 이미지 필터를 적용한 최종 결과
sobel_img = np.sqrt(np.square(horizontal_filtered) + np.square(vertical_filtered))

plt.imshow(sobel_img, cmap='gray')
plt.show()


# 3차원 데이터의 합성곱 연산
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img = url_to_image(img_url)
print('img shape:', img.shape)
plt.imshow(img)
plt.show()



image_copy = img.copy()
image_copy[:, :, 1] = 0
image_copy[:, :, 2] = 0
image_red = image_copy

print(image_red)
# [[[226   0   0]
#   [225   0   0]
#   [222   0   0]
#   ...
#   [233   0   0]
#   [224   0   0]
#   [202   0   0]]
# 
#  [[226   0   0]
#   [225   0   0]
#   [222   0   0]
#   ...
#   [233   0   0]
#   [224   0   0]
#   [202   0   0]]
#
#  [[226   0   0]
#   [225   0   0]
#   [222   0   0]
#   ...
#   [232   0   0]
#   [223   0   0]
#   [201   0   0]]
#
#  ...
# 
#  [[ 84   0   0]
#   [ 86   0   0]
#   [ 93   0   0]
#   ...
#   [174   0   0]
#   [169   0   0]
#   [172   0   0]]
# 
#  [[ 82   0   0]
#   [ 86   0   0]
#   [ 95   0   0]
#  ...
#   [177   0   0]
#   [178   0   0]
#   [183   0   0]]
# 
#  [[ 81   0   0]
#   [ 86   0   0]
#   [ 96   0   0]
#   ...
#   [178   0   0]
#   [181   0   0]
#   [185   0   0]]]



image_copy = img.copy()
image_copy[:, :, 0] = 0
image_copy[:, :, 2] = 0
image_green = image_copy



image_copy = img.copy()
image_copy[:, :, 0] = 0
image_copy[:, :, 1] = 0
image_blue = image_copy



fig = plt.figure(figsize=(12, 8))

title_list = ['R', 'G', 'B', 'R-grayscale', 'G-grayscale', 'B-grayscale']
image_list = [image_red, image_green, image_blue, image_red[:, :, 0], image_green[:, :, 1], image_green[:, :, 2]]

for i, image in enumerate(image_list):
    ax = fig.add_subplot(2, 3, i+1)
    ax.title.set_text('{}'.format(title_list[i]))

    if i >= 3:
      plt.imshow(image, cmap='gray')
    else:
      plt.imshow(image)

plt.show()





# util functions
def conv_op(image, kernel, pad=0, stride=1):
  H, W, C = image.shape
  kernel_size = kernel.shape[0]

  out_h = (H + 2*pad - kernel_size) // stride + 1
  out_w = (W + 2*pad - kernel_size) // stride + 1

  filtered_img = np.zeros((out_h, out_w))
  pad_img = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant')

  for i in range(out_h):
    for j in range(out_w):
      for c in range(C):
        multiply_values = pad_img[i*stride:(i*stride+kernel_size), j*stride:(j*stride+kernel_size)] * kernel
        sum_value = np.sum(multiply_values)
        filtered_img[i, j] = sum_value

  filtered_img = filtered_img.reshape(1, out_h, out_w, -1).transpose(0, 3, 1, 2)

  return filtered_img.astype(np.uint8)



# 이미지 확인
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img = url_to_image(img_url)
print('img shape:', img.shape)
plt.imshow(img)
plt.show()



# 필터연산 적용
filter1 = np.random.rand(3, 3, 3)
print(filter1.shape)
print(filter1)
# [[[0.03257119 0.06591546 0.32660524]
#   [0.51160199 0.40825431 0.72142329]
#   [0.2458546  0.53020835 0.70605638]]
#
# [[0.96408322 0.46880331 0.71743665]
#  [0.82851034 0.99134897 0.52593921]
#  [0.64543651 0.57735298 0.43106237]]
#
# [[0.59927753 0.8967411  0.28601834]
#  [0.85254497 0.38244151 0.33611396]
#  [0.67370228 0.52609544 0.54817692]]]



filtered_img1 = conv_op(img, filter1)
print(filtered_img1.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Used Filter')
plt.imshow(filter1, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Result')
plt.imshow(filtered_img1[0, 0, :, :], cmap='gray')
plt.show()



filter2 = np.random.rand(3, 3, 3)
print(filter2.shape)
print(filter2)
# [[[0.6168157  0.57754914 0.50569897]
#   [0.01206482 0.85905713 0.24357068]
#   [0.60398603 0.60965184 0.68755635]]
# 
#  [[0.98178854 0.82684043 0.8531998 ]
#   [0.21911992 0.67115191 0.9963988 ]
#   [0.54545965 0.80317275 0.95972389]]
#
#  [[0.65863854 0.26027196 0.74026522]
#   [0.86896287 0.14654397 0.1046968 ]
#   [0.00887471 0.23561989 0.54668264]]]



filtered_img2 = conv_op(img, filter2)
print(filtered_img2.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Used Filter')
plt.imshow(filter2, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Result')
plt.imshow(filtered_img2[0, 0, :, :], cmap='gray')
plt.show()