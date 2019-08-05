from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, Flatten, Lambda
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.losses import binary_crossentropy, kullback_leibler_divergence
from keras import backend as K
from struct import unpack
import matplotlib.pyplot as plt
import json
import glob
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
from io import BytesIO
import PIL
from PIL import ImageDraw
from keras.utils import get_file
from IPython.display import clear_output, Image, display, HTML
import os

def load_icons(path, train_size=0.85):
    x = []
    with open(path, 'rb') as f:
        while True:
            img = PIL.Image.new('L', (32, 32), 'white')
            draw = ImageDraw.Draw(img)
            header = f.read(15)
            if len(header) != 15:
                break
            strokes, = unpack('H', f.read(2))
            for i in range(strokes):
                n_points, = unpack('H', f.read(2))
                fmt = str(n_points) + 'B'
                read_scaled = lambda: (p // 8 for 
                                       p in unpack(fmt, f.read(n_points)))
                points = [*zip(read_scaled(), read_scaled())]
                draw.line(points, fill=0, width=2)
            img = img_to_array(img)
            x.append(img)
    x = np.asarray(x) / 255
    return train_test_split(x, train_size=train_size)


def create_autoencoder():
    input_img = Input(shape=(32, 32, 1))

    channels = 2
    x = input_img
    # 依次串联4个subnet，每个subnet包括两个卷积层+一个连接层+一个最大池化层
    for i in range(4):
        channels *= 2
        left = Conv2D(channels, (3, 3), activation='relu', padding='same')(x)
        right = Conv2D(channels, (2, 2), activation='relu', padding='same')(x)
        conc = Concatenate()([left, right])
        x = MaxPooling2D((2, 2), padding='same')(conc)
    
    # 一层全连接
    x = Dense(channels)(x)
    
    # 依次串联4个双层，每个双层包括一层卷积一层上采样（类似于逆向池化）。
    for i in range(4):
        x = Conv2D(channels, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        channels //= 2
    
    # 最后再来一层卷积层
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder


# 载入数据，需要从这个链接下载：https://storage.googleapis.com/quickdraw_dataset/full/binary/cat.bin。
# BASE_PATH = 'https://storage.googleapis.com/quickdraw_dataset/full/binary/'
# path = get_file('cat', BASE_PATH + 'cat.bin')
path = os.getcwd()+'/data/cat.bin'
x_train, x_test = load_icons(path)
# 形状为(104721, 32, 32, 1), (18481, 32, 32, 1)
print(x_train.shape, x_test.shape)


autoencoder = create_autoencoder()
autoencoder.summary()
""" 编码器网络的结构说明
注意：
1.有参数的层：卷积层，全连接层；
2.没有参数的层：输入层，连接层，池化层，上采样层。
3.每层名字的后缀数字，表示同类型层的序号。
4.总共有14层网络具备参数：13个卷积层+1个全连接层

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 1)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 4)    40          input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 4)    20          input_1[0][0]                    
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 32, 32, 8)    0           conv2d_1[0][0]                   
                                                                 conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 16, 16, 8)    0           concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 16, 16, 8)    584         max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 16, 16, 8)    264         max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 16, 16, 16)   0           conv2d_3[0][0]                   
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 8, 8, 16)     0           concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 8, 8, 16)     2320        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 8, 8, 16)     1040        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 8, 8, 32)     0           conv2d_5[0][0]                   
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 4, 4, 32)     0           concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 4, 32)     9248        max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 4, 4, 32)     4128        max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 4, 4, 64)     0           conv2d_7[0][0]                   
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 2, 2, 64)     0           concatenate_4[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2, 2, 32)     2080        max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 2, 2, 32)     9248        dense_1[0][0]                    
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 4, 4, 32)     0           conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 4, 4, 16)     4624        up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 8, 8, 16)     0           conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 8, 8, 8)      1160        up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 16, 16, 8)    0           conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 4)    292         up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)  (None, 32, 32, 4)    0           conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 32, 32, 1)    37          up_sampling2d_4[0][0]            
==================================================================================================
Total params: 35,085
Trainable params: 35,085
Non-trainable params: 0
__________________________________________________________________________________________________
"""
