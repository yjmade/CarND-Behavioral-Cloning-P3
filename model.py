# -*- coding: utf-8 -*-
# import csv

# data_path = ["data/driving_log.csv", "data/driving_log2.csv"]
# parent_folder = "data/IMG/"
# with open(data_path) as f:
#     for line in csv.

import pandas as pd
import numpy as np
import os
from PIL import Image
import random
import glob

# X_train = np.array(
#     [np.asarray(Image.open(get_image_path(path))) for path in data["center"]] +
#     [np.asarray(Image.open(get_image_path(path))) for path in data["left"]] +
#     [np.asarray(Image.open(get_image_path(path))) for path in data["right"]]
# )

# y_train = data["steer"].as_matrix()
# y_train = np.append(y_train, data["steer"].as_matrix() + 0.2)
# y_train = np.append(y_train, data["steer"].as_matrix() - 0.2)

funcs = [
    ("center", lambda x, y:(x, y)),
    ("left", lambda x, y:(x, y + 0.2)),
    ("right", lambda x, y:(x, y - 0.2)),
]


flip = [
    lambda x, y:(x, y),
    lambda x, y:(x[:, ::-1], -y)
]

data = []
for path in glob.iglob("/home/yjmade/Documents/*/*.csv"):
    img_folder = os.path.join(os.path.dirname(path), "IMG")
    csv_data = pd.read_csv(path, names=["center", "left", "right", "steer", "thottle", "break", "_"]).as_matrix()
    data.append(np.concatenate([csv_data,np.reshape([img_folder]*csv_data.shape[0],[-1,1])],axis=1))
data = np.concatenate(data, axis=0)
np.random.shuffle(data)
a_data = data
data, valid_data = data[:int(len(data) * 0.9)], data[int(len(data) * 0.9):]
print("Training data size:", data.shape[0])

def gen_each_data(item):
    pos, func = random.choice(funcs)
    img = np.asarray(Image.open(get_image_path(item["img_folder"], item[pos])))
    X, y = func(img, item["steer"])
    X, y = random.choice(flip)(X, y)
    return X, y

# @threadsafe_generator


def gen(data, batch_size):
    # data.apply(np.random.shuffle, axis=0)
    i = -1
    total_count = len(data)
    # g = Parallel(n_jobs=1, max_nbytes=None)
    while True:
        i += 1
        xs = []
        ys = []

        # for X, y in g(
        #     delayed(gen_each_data)(dict(zip(["center", "left", "right", "steer", "thottle", "break", "_"], datassssss[i % total_count])))
        #     for i in range(i, i + batch_size)
        # ):
        for X, y in (
            gen_each_data(dict(zip(["center", "left", "right", "steer", "thottle", "break", "_", "img_folder"], data[i % total_count])))
            for i in range(i, i + batch_size)
        ):
            xs.append(X)
            ys.append(y)
        yield np.array(xs), np.array(ys)


def get_image_path(parent_folder, old_path):
    return os.path.join(parent_folder, os.path.split(old_path)[-1])


# flip
# X_train = np.append(X_train, X_train[:, :, ::-1], axis=0)
# y_train = np.append(y_train, -y_train)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, Cropping2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: (x - 128.) / 128., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=[(70, 25), (0, 0)]))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.7))
model.add(Dense(50))
model.add(Dropout(.6))
model.add(Dense(10))
model.add(Dropout(.5))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
# model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1, callbacks=None, validation_split=0.2)
batch_size = 256
if __name__ == "__main__":
    model.fit_generator(
        gen(data, batch_size),
        steps_per_epoch=3 * len(data) / batch_size,
        nb_epoch=10,
        validation_data=gen(valid_data, batch_size),
        validation_steps=5,
        verbose=1,
        # workers=10
    )
    model.save("model.h5")
