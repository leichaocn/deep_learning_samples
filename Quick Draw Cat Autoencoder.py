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

from keras.callbacks import TensorBoard

# 训练100轮，大概数小时。如果是cpu的话。
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# 随机取25张测试图片，进行编码解码，并查看解码后的输出形状
cols = 25
idx = np.random.randint(x_test.shape[0], size=cols)
sample = x_test[idx]
decoded_imgs = autoencoder.predict(sample)
# 形状为(25, 32, 32, 1)
print(decoded_imgs.shape)


# 横排显示25张输入图片与输出图片的对比
def decode_img(tile, factor=1.0):
    tile = tile.reshape(tile.shape[:-1])
    tile = np.clip(tile * 255, 0, 255)
    return PIL.Image.fromarray(tile)
    

overview = PIL.Image.new('RGB', (cols * 32, 64 + 20), (128, 128, 128))
for idx in range(cols):
    overview.paste(decode_img(sample[idx]), (idx * 32, 5))
    overview.paste(decode_img(decoded_imgs[idx]), (idx * 32, 42))
f = BytesIO()
overview.save(f, 'png')
display(Image(data=f.getvalue()))






# 
batch_size = 250
latent_space_depth = 64
def sample_z(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(batch_size, latent_space_depth), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * eps

# 构造变分自编码器
def VariationalAutoEncoder(num_pixels):
    """变分自编码器结构说明：
        1.先5个串联block，每个block=2个卷积+1层连接+1层最大池化
        2.然后是全连接+flatten+全连接+全连接+lambda_1+reshape
        3.然后是5个block，每个block=1层卷积+1层上采样
        4.最后一层是卷积层
    """
    
    input_img = Input(shape=(32, 32, 1))

    channels = 4
    x = input_img
    # 先5个串联block，每个block=2个卷积+1层连接+1层最大池化
    for i in range(5):
        left = Conv2D(channels, (3, 3), activation='relu', padding='same')(x)
        right = Conv2D(channels, (2, 2), activation='relu', padding='same')(x)
        conc = Concatenate()([left, right])
        x = MaxPooling2D((2, 2), padding='same')(conc)
        channels *= 2

    # 然后是全连接+flatten+全连接+全连接+lambda_1+reshape
    x = Dense(channels)(x)
    encoder_hidden = Flatten()(x)

    z_mean = Dense(latent_space_depth, activation='linear')(encoder_hidden)
    z_log_var = Dense(latent_space_depth, activation='linear')(encoder_hidden)
    
    def KL_loss(y_true, y_pred):
        return 0.0001 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1 - z_log_var, axis=1)

    def reconstruction_loss(y_true, y_pred):
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        return binary_crossentropy(y_true, y_pred)

    def total_loss(y_true, y_pred):
        return reconstruction_loss(y_true, y_pred) + KL_loss(y_true, y_pred)

    z = Lambda(sample_z, output_shape=(latent_space_depth, ))([z_mean, z_log_var])
    decoder_in = Input(shape=(latent_space_depth,))

    d_x = Reshape((1, 1, latent_space_depth))(decoder_in)
    e_x = Reshape((1, 1, latent_space_depth))(z)
    
    # 然后是5个block，每个block=1层卷积+1层上采样
    for i in range(5):
        conv = Conv2D(channels, (3, 3), activation='relu', padding='same')
        upsampling = UpSampling2D((2, 2))
        d_x = conv(d_x)
        d_x = upsampling(d_x)
        e_x = conv(e_x)
        e_x = upsampling(e_x)
        channels //= 2
    
    # 最后一层是卷积层
    final_conv = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    auto_decoded = final_conv(e_x)
    decoder_out = final_conv(d_x)
    
    decoder = Model(decoder_in, decoder_out)    
    
    auto_encoder = Model(input_img, auto_decoded)

    auto_encoder.compile(optimizer=Adam(lr=0.001), 
                         loss=total_loss,
                         metrics=[KL_loss, reconstruction_loss])
    
    return auto_encoder, decoder

variational_auto_encoder, variational_decoder = VariationalAutoEncoder(x_train.shape[1])
variational_auto_encoder.summary()
"""变分自编码器结构说明：
1.先5个串联block，每个block=2个卷积+1层连接+1层最大池化
2.然后是全连接+flatten+全连接+全连接+lambda_1+reshape
3.然后是5个block，每个block=1层卷积+1层上采样
4.最后一层是卷积层

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 32, 32, 1)    0                                            
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 4)    40          input_2[0][0]                    
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 32, 32, 4)    20          input_2[0][0]                    
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 32, 32, 8)    0           conv2d_14[0][0]                  
                                                                 conv2d_15[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 16, 16, 8)    0           concatenate_5[0][0]              
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 16, 16, 8)    584         max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 16, 16, 8)    264         max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 16, 16, 16)   0           conv2d_16[0][0]                  
                                                                 conv2d_17[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 8, 8, 16)     0           concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 8, 8, 16)     2320        max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 8, 8, 16)     1040        max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 8, 8, 32)     0           conv2d_18[0][0]                  
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 4, 4, 32)     0           concatenate_7[0][0]              
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 4, 4, 32)     9248        max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 4, 4, 32)     4128        max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 4, 4, 64)     0           conv2d_20[0][0]                  
                                                                 conv2d_21[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 2, 2, 64)     0           concatenate_8[0][0]              
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 2, 2, 64)     36928       max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 2, 2, 64)     16448       max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 2, 2, 128)    0           conv2d_22[0][0]                  
                                                                 conv2d_23[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_9 (MaxPooling2D)  (None, 1, 1, 128)    0           concatenate_9[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1, 1, 128)    16512       max_pooling2d_9[0][0]            
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 128)          0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 64)           8256        flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 64)           8256        flatten_1[0][0]                  
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 64)           0           dense_3[0][0]                    
                                                                 dense_4[0][0]                    
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 1, 1, 64)     0           lambda_1[0][0]                   
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 1, 1, 128)    73856       reshape_2[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)  (None, 2, 2, 128)    0           conv2d_24[1][0]                  
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 2, 2, 64)     73792       up_sampling2d_5[1][0]            
__________________________________________________________________________________________________
up_sampling2d_6 (UpSampling2D)  (None, 4, 4, 64)     0           conv2d_25[1][0]                  
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 4, 4, 32)     18464       up_sampling2d_6[1][0]            
__________________________________________________________________________________________________
up_sampling2d_7 (UpSampling2D)  (None, 8, 8, 32)     0           conv2d_26[1][0]                  
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 8, 8, 16)     4624        up_sampling2d_7[1][0]            
__________________________________________________________________________________________________
up_sampling2d_8 (UpSampling2D)  (None, 16, 16, 16)   0           conv2d_27[1][0]                  
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 16, 16, 8)    1160        up_sampling2d_8[1][0]            
__________________________________________________________________________________________________
up_sampling2d_9 (UpSampling2D)  (None, 32, 32, 8)    0           conv2d_28[1][0]                  
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 32, 32, 1)    73          up_sampling2d_9[1][0]            
==================================================================================================
Total params: 276,013
Trainable params: 276,013
Non-trainable params: 0
__________________________________________________________________________________________________
"""

# 数据准备
x_train_2 = x_train[:-(x_train.shape[0] % batch_size),:,: :]
x_test_2 = x_test[:-(x_test.shape[0] % batch_size),:,: :]
# 形状为((104500, 32, 32, 1), (18250, 32, 32, 1))
print(x_train_2.shape, x_test_2.shape)

# 变分自编码器的训练，约数小时。
variational_auto_encoder.fit(x_train_2, x_train_2, verbose=1, 
                 batch_size=batch_size, epochs=100,
                 validation_data=(x_test_2, x_test_2))


# 生成一个形状为（1，latent_space_depth）的高斯随机数组random_number
random_number = np.asarray([[np.random.normal() 
                            for _ in range(latent_space_depth)]])
img_width, img_height = 32, 32
def decode_img(a):
    a = np.clip(a * 256, 0, 255).astype('uint8')
    return PIL.Image.fromarray(a)

decode_img(variational_decoder.predict(random_number).reshape(img_width, img_height))


# 展示10*10的矩阵生成图
num_cells = 10
img_width = img_height = 32
overview = PIL.Image.new('RGB', 
                         (num_cells * (img_width + 4) + 8, 
                          num_cells * (img_height + 4) + 8), 
                         (140, 128, 128))

for x in range(num_cells):
    for y in range(num_cells):
        vec = np.asarray([[np.random.normal() 
                            for _ in range(latent_space_depth)]])
        decoded = variational_decoder.predict(vec)
        img = decode_img(decoded.reshape(img_width, img_height))
        overview.paste(img, (x * (img_width + 4) + 6, y * (img_height + 4) + 6))
overview


