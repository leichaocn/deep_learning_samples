"""
# 首先需要安装古藤包语料库
"""
try:
    GUTENBERG = True
    from gutenberg.acquire import load_etext
    from gutenberg.query import get_etexts, get_metadata
    from gutenberg.acquire import get_metadata_cache
    from gutenberg.acquire.text import UnknownDownloadUriException
    from gutenberg.cleanup import strip_headers
    from gutenberg._domain_model.exceptions import CacheAlreadyExistsException
except ImportError:
    GUTENBERG = False
    print('Gutenberg is not installed. See instructions at https://pypi.python.org/pypi/Gutenberg')
from keras.models import Input, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
import keras.callbacks
import keras.backend as K
import scipy.misc
import json

import os, sys
import re
import PIL
from PIL import ImageDraw

from keras.optimizers import RMSprop
import random
import numpy as np
import tensorflow as tf
from keras.utils import get_file

from IPython.display import clear_output, Image, display, HTML
try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

if GUTENBERG:
    cache = get_metadata_cache()
    try:
        cache.populate()
    except CacheAlreadyExistsException:
        pass
"""
查看作品名
"""
if GUTENBERG:
    for text_id in get_etexts('author', 'Shakespeare, William'):
        print(text_id, list(get_metadata('title', text_id))[0])

"""
抽出其中一篇作为训练语料
"""
if GUTENBERG:
    shakespeare = strip_headers(load_etext(100))
else:
    path = get_file('shakespeare', 'https://storage.googleapis.com/deep-learning-cookbook/100-0.txt')
    shakespeare = open(path).read()
training_text = shakespeare.split('\nTHE END', 1)[-1]
print(len(training_text))
print(training_text[:1000])


"""
抽出所有类型的单字符，并编号。
chars = ['\t', '\n', ' ', '!', '"', '&', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '}', 'Æ', 'É', 'à', 'â', 'æ', 'ç', 'è', 'é', 'ê', 'î', 'œ', '—', '‘', '’', '“', '”']
char_to_idx = {'X': 49, 'a': 57, 'æ': 89, 'U': 46, 'K': 36, 'm': 69, 'E': 30, 'R': 43, 'P': 41, 'é': 92, 'à': 87, '‘': 97, 'B': 27, '}': 84, '0': 13, '\n': 1, 'N': 39, '.': 12, '2': 15, '6': 19, 'y': 81, 'p': 72, 'b': 58, '\t': 0, ':': 23, "'": 6, ';': 24, 'I': 34, '`': 56, '’': 98, 'è': 91, '8': 21, '“': 99, 'L': 37, 'c': 59, 'S': 44, '—': 96, '\\': 53, '5': 18, 't': 76, 'x': 80, 'Y': 50, 'É': 86, 'Æ': 85, 'q': 73, 'ç': 90, 'F': 31, 'O': 40, 'A': 26, 'k': 67, '9': 22, 'z': 82, 'e': 61, ',': 10, 'J': 35, 'W': 48, 'ê': 93, 'j': 66, 'Q': 42, 'v': 78, 'u': 77, 'M': 38, '4': 17, 'T': 45, ' ': 2, 'h': 64, 'l': 68, '*': 9, 'D': 29, '?': 25, 'g': 63, '!': 3, 'f': 62, '1': 14, ')': 8, 'H': 33, '&': 5, '"': 4, 'Z': 51, '7': 20, ']': 54, 'd': 60, 'w': 79, '”': 100, 'V': 47, 'œ': 95, '(': 7, 'G': 32, 'â': 88, '|': 83, '-': 11, '_': 55, 's': 75, 'n': 70, '3': 16, 'î': 94, 'i': 65, 'r': 74, 'o': 71, '[': 52, 'C': 28}
"""
chars = list(sorted(set(training_text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
print(chars)
print(len(chars))
print(char_to_idx)

"""
构造RNN模型
"""
def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
    input = Input(shape=(None, num_chars), name='input')
    prev = input
    for i in range(num_layers):
        lstm = LSTM(num_nodes, return_sequences=True, name='lstm_layer_%d' % (i + 1))(prev)
        if dropout:
            prev = Dropout(dropout)(lstm)
        else:
            prev = lstm
    dense = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(prev)
    model = Model(inputs=[input], outputs=[dense])
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    
model = char_rnn_model(len(chars), num_layers=2, num_nodes=640, dropout=0)
model.summary()
"""网络结构如下：
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           (None, None, 101)         0         
_________________________________________________________________
lstm_layer_1 (LSTM)          (None, None, 640)         1899520   
_________________________________________________________________
lstm_layer_2 (LSTM)          (None, None, 640)         3279360   
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 101)         64741     
=================================================================
Total params: 5,243,621
Trainable params: 5,243,621
Non-trainable params: 0
_________________________________________________________________
"""

"""

"""
CHUNK_SIZE = 160
def data_generator(all_text, char_to_idx, batch_size, chunk_size):
    X = np.zeros((batch_size, chunk_size, len(char_to_idx)))
    y = np.zeros((batch_size, chunk_size, len(char_to_idx)))
    while True:
        for row in range(batch_size):
            idx = random.randrange(len(all_text) - chunk_size - 1)
            chunk = np.zeros((chunk_size + 1, len(char_to_idx)))
            for i in range(chunk_size + 1):
                chunk[i, char_to_idx[all_text[idx + i]]] = 1
            X[row, :, :] = chunk[:chunk_size]
            y[row, :, :] = chunk[1:]
        yield X, y

next(data_generator(training_text, char_to_idx, 4, chunk_size=CHUNK_SIZE))

"""
迭代式训练？
"""
early = keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=0.03,
                              patience=3,
                              verbose=0, mode='auto')

BATCH_SIZE = 256
model.fit_generator(
    data_generator(training_text, char_to_idx, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE),
    epochs=40,
    callbacks=[early,],
    steps_per_epoch=2 * len(training_text) / (BATCH_SIZE * CHUNK_SIZE),
    verbose=2
)
