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
# output size: 438



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
# img shape: (440, 440, 3)



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
# (3, 3, 3)
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
# (1, 1, 438, 438)



filter2 = np.random.rand(3, 3, 3)
print(filter2.shape)
print(filter2)
# (3, 3, 3)
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
# (1, 1, 438, 438)



filter3 = np.random.rand(3, 3, 3)
print(filter3.shape)
print(filter3)
# (3, 3, 3)
# [[[0.0059815  0.40140623 0.83091443]
#   [0.19433548 0.49284306 0.95681345]
#   [0.77810437 0.75658527 0.55424864]]
# 
#  [[0.49316376 0.85017094 0.81019837]
#   [0.34103453 0.83804446 0.17615835]
#   [0.03765345 0.91629288 0.84902376]]
# 
#  [[0.38181276 0.36307186 0.36201367]
#   [0.63322291 0.71188052 0.28136281]
#   [0.69190981 0.62155285 0.42441674]]]



filtered_img3 = conv_op(img, filter3)
print(filtered_img3.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Used Filter')
plt.imshow(filter3, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Result')
plt.imshow(filtered_img3[0, 0, :, :], cmap='gray')
plt.show()
# (1, 1, 438, 438)



filter4 = np.random.rand(3, 3, 3)
print(filter4.shape)
print(filter4)
# (3, 3, 3)
# [[[0.54115642 0.3707889  0.08661744]
#   [0.90570731 0.86453098 0.9250265 ]
#   [0.31361625 0.60145525 0.34792101]]
# 
#  [[0.17682912 0.86697906 0.94455943]
#   [0.30568459 0.07742387 0.40111568]
#   [0.35756594 0.96758562 0.70123002]]
# 
#  [[0.91553969 0.07810065 0.88000496]
#   [0.78497317 0.53876645 0.13940625]
#   [0.05290455 0.77399082 0.3503566 ]]]



filtered_img4 = conv_op(img, filter4)
print(filtered_img4.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Used Filter')
plt.imshow(filter4, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Result')
plt.imshow(filtered_img4[0, 0, :, :], cmap='gray')
plt.show()
# (1, 1, 438, 438)



filter5 = np.random.rand(3, 3, 3)
print(filter5.shape)
print(filter5)
# (3, 3, 3)
# [[[0.46738546 0.87800265 0.77300912]
#   [0.80673424 0.35636831 0.74340681]
#   [0.51509229 0.76863137 0.01697408]]
# 
#  [[0.51379273 0.11327817 0.9657751 ]
#   [0.83649732 0.87781192 0.07331755]
#   [0.26067299 0.07345458 0.89057797]]
# 
#  [[0.05774411 0.54606462 0.95869447]
#   [0.20710387 0.83927981 0.66621642]
#   [0.21580274 0.24732934 0.28040942]]]



filtered_img5 = conv_op(img, filter5)
print(filtered_img5.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Used Filter')
plt.imshow(filter5, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Result')
plt.imshow(filtered_img5[0, 0, :, :], cmap='gray')
plt.show()
# (1, 1, 438, 438)



# 필터연산을 적용한 최종 결과
filtered_img = np.stack([filtered_img1, filtered_img2, filtered_img3, filtered_img4, filtered_img5]).sum(axis=0)

print(filtered_img.shape)
plt.imshow(filtered_img[0, 0, :, :], cmap='gray')
plt.show()
# (1, 1, 438, 438)



# 전체 과정 한번에 보기
np.random.seed(222)

fig = plt.figure(figsize=(8, 20))

filter_num = 5
filtered_img = []

for i in range(filter_num):
    ax = fig.add_subplot(5, 2, 2*i+1)
    ax.title.set_text('Filter {}'.format(filter_num+1))

    filter = np.random.randn(3, 3, 3)
    plt.imshow(filter)

    ax = fig.add_subplot(5, 2, 2*i+2)
    ax.title.set_text('Result')

    filtered = conv_op(img, filter)
    filtered_img.append(filtered)
    plt.imshow(filtered[0, 0, :, :], cmap='gray')

plt.show()
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.6407514495107147..1.9634250169646124].
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7201688082560667..1.9925713052811553].
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.039940178522755..2.441719813587031].
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7198605830142413..1.9994749849283326].
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.3136237868316294..1.8832216580390868].



filter
# array([[[ 0.83743502, -1.45547212, -0.38195419],
#         [-2.31362379,  0.76470662, -1.16146643],
#         [ 0.45189131,  0.5713418 ,  0.56447011]],
# 
#        [[ 0.1999429 , -0.48757907,  1.88322166],
#         [ 0.99848839, -0.2407538 , -0.69350822],
#         [-0.26178885,  0.21134661,  0.39584267]],
# 
#        [[ 0.48039944, -0.36013447, -1.40382323],
#         [-2.255036  ,  1.2505411 , -0.21994389],
#         [ 1.19205002, -0.62169983, -1.98899032]]])



filtered_img = np.stack(filtered_img).sum(axis=0)

print(filtered_img.shape)
plt.imshow(filtered_img[0, 0, :, :], cmap='gray')
plt.show()
# (1, 1, 438, 438)



# 합성곱 신경망 구현
# 합성곱 층(Convolution Layer)
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



class Conv2D:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.input_data = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, input_data):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1

        col = im2col(input_data, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.input_data = input_data
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.input_data.shape, FH, FW, self.stride, self.pad)



# 컨볼루션 레이어 테스트
def init_weight(num_filters, data_dim, kernel_size, stride=1, pad=0, weight_std=0.01):
    weights = np.random.randn(num_filters, data_dim, kernel_size, kernel_size) * weight_std
    biases = np.zeros(num_filters)

    return weights, biases



img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img_gray = url_to_image(img_url, gray=True)
img_gray = img_gray.reshape(img_gray.shape[0], -1, 1)
print('img shape:', img_gray.shape)

img_gray = np.expand_dims(img_gray.transpose(2, 0, 1), axis=0)

plt.imshow(img_gray[0, 0, :, :], cmap='gray')
plt.show()
# img shape: (440, 440, 1)



W, b = init_weight(1, 1, 3)
conv = Conv2D(W, b)
output = conv.forward(img_gray)

print('Conv Layer size:', output.shape)
# Conv Layer size: (1, 1, 438, 438)



img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img_gray = url_to_image(img_url, gray=True)
img_gray = img_gray.reshape(img_gray.shape[0], -1, 1)
print('img shape:', img_gray.shape)
img_gray = np.expand_dims(img_gray.transpose(2, 0, 1), axis=0)
plt.imshow(img_gray[0, 0, :, :], cmap='gray')
plt.show()
# img shape: (440, 440, 1)



W, b = init_weight(1, 1, 3)
conv = Conv2D(W, b)
output = conv.forward(img_gray)

print('Conv Layer size:', output.shape)
# Conv Layer size: (1, 1, 438, 438)



plt.imshow(output[0, 0, :, :], cmap='gray')
plt.show()



W2, b2 = init_weight(1, 1, 3, stride=2)
conv2 = Conv2D(W2, b2, stride=2)
output2 = conv2.forward(img_gray)

print('Conv Layer size:', output2.shape)
# Conv Layer size: (1, 1, 219, 219)



plt.imshow(output2[0, 0, :, :], cmap='gray')
plt.show()



img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img = url_to_image(img_url)
print('img shape:', img.shape)
plt.imshow(img)
plt.show()

img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
print('img.shape:', img.shape)
# img shape: (440, 440, 3)
# img.shape: (1, 3, 440, 440)



W3, b3 = init_weight(10, 3, 3)
conv3 = Conv2D(W3, b3)
output3 = conv3.forward(img)

print('Conv Layer size:', output3.shape)
# Conv Layer size: (1, 10, 438, 438)



plt.imshow(output3[0, 3, :, :], cmap='gray')
plt.show()



plt.imshow(output3[0, 0, :, :], cmap='gray')
plt.show()



# 동일한 이미지 여러 장 테스트 (배치 처리)
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img_gray = url_to_image(img_url, gray=True)
img_gray = img_gray.reshape(img_gray.shape[0], -1, 1)
print('img shape:', img_gray.shape)

img_gray = img_gray.transpose(2, 0, 1)
print('img_gray.shape:', img_gray.shape)
# img shape: (440, 440, 1)
# img_gray.shape: (1, 440, 440)



batch_img_gray = np.repeat(img_gray[np.newaxis, :, :, :], 15, axis=0)
print(batch_img_gray.shape)
# (15, 1, 440, 440)



W4, b4 = init_weight(10, 1, 3, stride=2)
conv4 = Conv2D(W4, b4, stride = 2)
output4 = conv4.forward(batch_img_gray)

print('Conv Layer size:', output4.shape)
# Conv Layer size: (15, 10, 219, 219)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Filter 3')
plt.imshow(output4[3, 2, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Filter 6')
plt.imshow(output4[3, 5, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Filter 10')
plt.imshow(output4[3, 9, :, :], cmap='gray')

plt.show()



W5, b5 = init_weight(32, 3, 3, stride=3)
conv5 = Conv2D(W5, b5, stride = 3)
output5 = conv5.forward(img)

print('Conv Layer size:', output5.shape)
# Conv Layer size: (1, 32, 146, 146)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Filter 21')
plt.imshow(output5[0, 20, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Filter 15')
plt.imshow(output5[0, 14, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Filter 11')
plt.imshow(output5[0, 10, :, :], cmap='gray')

plt.show()



# 동일한 이미지 배치처리(color)
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img = url_to_image(img_url)
print('img shape:', img.shape)
plt.imshow(img)
plt.show()
# img shape: (440, 440, 3)


img = img.transpose(2, 0, 1)
print('img.shape:', img.shape)
# img.shape: (3, 440, 440)



batch_image_color = np.repeat(img[np.newaxis, :, :, :], 15, axis=0)
print(batch_image_color.shape)
# (15, 3, 440, 440)



W6, b6 = init_weight(64, 3, 5)
conv6 = Conv2D(W6, b6)
output6 = conv6.forward(batch_image_color)

print('Conv Layer size:', output6.shape)
# Conv Layer size: (15, 64, 436, 436)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Filter 50')
plt.imshow(output6[10, 49, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Filter 31')
plt.imshow(output6[10, 30, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Filter 1')
plt.imshow(output6[10, 0, :, :], cmap='gray')

plt.show()



# 풀링 층
class Pooling2D:
    def __init__(self, kernel_size=2, stride=1, pad=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        self.input_data = None
        self.arg_max = None

    def forward(self, input_data):
        N, C, H, W = input_data.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        col = im2col(input_data, self.kernel_size, self.kernel_size, self.stride, self.pad)
        col = col.reshape(-1, self.kernel_size*self.kernel_size)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        output = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.input_data = input_data
        self.arg_max = arg_max

        return output

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.kernel_size * self.kernel_size
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.input_data.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)

        return dx
    


# 풀링 레이어 테스트
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img_gray = url_to_image(img_url, gray=True)
img_gray = img_gray.reshape(img_gray.shape[0], -1, 1)
print('img shape:', img_gray.shape)
img_gray = np.expand_dims(img_gray.transpose(2, 0, 1), axis=0)
plt.imshow(img_gray[0, 0, :, :], cmap='gray')
plt.show()
# img shape: (440, 440, 1)



W, b = init_weight(8, 1, 3)
conv = Conv2D(W, b)
pool = Pooling2D(stride=2, kernel_size=2)
output1 = conv.forward(img_gray)

print('Conv Layer size:', output1.shape)
# Conv Layer size: (1, 8, 438, 438)



output1 = pool.forward(output1)
print('Pooling Layer size:', output1.shape)
# Pooling Layer size: (1, 8, 219, 219)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Feature Map 8')
plt.imshow(output1[0, 7, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Feature Map 4')
plt.imshow(output1[0, 3, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Feature Map 1')
plt.imshow(output1[0, 0, :, :], cmap='gray')

plt.show()



W2, b2 = init_weight(32, 1, 3, stride=2)
conv2 = Conv2D(W2, b2)
pool = Pooling2D(kernel_size=2)
output2 = conv2.forward(img_gray)
output2 = pool.forward(output2)

print('Conv Layer size:', output2.shape)
# Conv Layer size: (1, 32, 437, 437)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Feature Map 8')
plt.imshow(output2[0, 7, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Feature Map 4')
plt.imshow(output2[0, 3, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Feature Map 1')
plt.imshow(output2[0, 0, :, :], cmap='gray')

plt.show()



# 동일한 이미지 배치처리
img_url = 'https://upload.wikimedia.org/wikipedia/ko/thumb/2/24/Lenna.png/440px-Lenna.png'

img = url_to_image(img_url)
print('img shape:', img.shape)
plt.imshow(img)
plt.show()
# img shape: (440, 440, 3)



img = img.transpose(2, 0, 1)
print('img.shape:', img.shape)
# img.shape: (3, 440, 440)



batch_image_color = np.repeat(img[np.newaxis, :, :, :], 15, axis=0)
print(batch_image_color.shape)
# (15, 3, 440, 440)



W, b = init_weight(10, 3, 3)
conv1 = Conv2D(W, b)
pool = Pooling2D(stride=2, kernel_size=2)
output1 = conv1.forward(batch_image_color)

print('Conv Layer size:', output1.shape)
# Conv Layer size: (15, 10, 438, 438)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Featuer Map 2')
plt.imshow(output1[4, 1, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Featuer Map 5')
plt.imshow(output1[4, 4, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Featuer Map 9')
plt.imshow(output1[4, 8, :, :], cmap='gray')

plt.show()



output1 = pool.forward(output1)
print('Pooling Layer size:', output1.shape)
# Pooling Layer size: (15, 10, 219, 219)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Featuer Map 2')
plt.imshow(output1[4, 1, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Featuer Map 5')
plt.imshow(output1[4, 4, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Featuer Map 9')
plt.imshow(output1[4, 8, :, :], cmap='gray')

plt.show()



W2, b2 = init_weight(30, 10, 3)
conv2 = Conv2D(W2, b2)
pool = Pooling2D(stride=2, kernel_size=2)
output2 = conv2.forward(output1)
# output2 = pool.forward(output2)

print('Conv Layer size:', output2.shape)
# Conv Layer size: (15, 30, 217, 217)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Featuer Map 2')
plt.imshow(output2[4, 1, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Featuer Map 5')
plt.imshow(output2[4, 4, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Featuer Map 9')
plt.imshow(output2[4, 8, :, :], cmap='gray')

plt.show()



output2 = pool.forward(output2)

print('Conv Layer size:', output2.shape)
# Conv Layer size: (15, 30, 108, 108)



plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Featuer Map 2')
plt.imshow(output2[4, 1, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Featuer Map 5')
plt.imshow(output2[4, 4, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Featuer Map 9')
plt.imshow(output2[4, 8, :, :], cmap='gray')

plt.show()