import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras import backend as K
import os
import numpy as np
import json
from collections import Counter
from keras.optimizers import SGD

# 图片存放路径。数据下载地址：http://www.robots.ox.ac.uk/~vgg/data/pets/
images_path = os.getcwd()+'/data/pet_images'


def fetch_pet(pet):
    img = image.load_img(os.getcwd()+'/data/pet_images/' + pet, target_size=(299, 299))
    return image.img_to_array(img)


### 数据组装 
pet_images_fn = [fn for fn in os.listdir(images_path) if fn.endswith('.jpg')]
# label数组，用来按照图片顺序，依次装入数字label。
labels = []
# 由数字到文字label的数组
idx_to_labels = []
# 由文字label到数字的字典
label_to_idx = {}
for fn in pet_images_fn:
    # rsplit分隔方法，从后往前，分隔1次，用下划线分隔，相当于抽取出了label。
    label, _ = fn.rsplit('_', 1)
    if not label in label_to_idx:
        # 每个label对应的数字其实很随意生成的，只要对应一个特殊的数字即可。注意，这是字典
        label_to_idx[label] = len(idx_to_labels)
        # 紧接着，就把这个文字label添加到数组的这个索引位，该索引位正是
        idx_to_labels.append(label)
    # 依次装入数字label
    labels.append(label_to_idx[label])
# len(idx_to_labels)

img_vector = np.asarray([fetch_pet(pet) for pet in pet_images_fn])
###


### 预训练模型的下载与结构更改
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in base_model.layers:
    layer.trainable = False
pool_2d = GlobalAveragePooling2D(name='pool_2d')(base_model.output)
dense = Dense(1024, name='dense', activation='relu')(pool_2d)
predictions = Dense(len(idx_to_labels), activation='softmax')(dense)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


### 模型训练
y = np.zeros((len(labels), len(idx_to_labels)))
for idx, label in enumerate(labels):
    y[idx][label] = 1
model.fit(
    img_vector, y,
    batch_size=128,
    epochs=15,
    verbose=2
)


unfreeze = False
for layer in base_model.layers:
    if unfreeze:
        layer.trainable = True
    if layer.name == 'mixed9':
        unfreeze = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    img_vector, y,
    batch_size=128,
    epochs=10,
    verbose=2
)

### 模型的保存
json.dump(idx_to_labels, open('zoo/09.3 pet_labels.json', 'w'))
model.save('zoo/09.3 retrained pet recognizer.h5')
