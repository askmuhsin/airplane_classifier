import glob
import time
import pickle
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

## --------------------------------------------

negative_dir = glob.glob('dataset/negative/*.png')
positive_dir = glob.glob('dataset/positive/*.png')

plane = []
not_plane = []
label = []

for image in negative_dir:
    not_plane.append(image)
    label.append(0)

for image in positive_dir:
    plane.append(image)
    label.append(1)

print("# of not_plane:\t", len(not_plane))
print("# of plane:\t", len(plane))
## --------------------------------------------

X = not_plane + plane
y = np.array(label)
y = y.reshape(y.shape[0],1)

assert len(X)==len(y), "mismatch in data and label size"

# ## Exploratory data
# for _ in range(5):
#     n = random.randint(0, len(X)-1)
#     img_ = plt.imread(X[n])
#     label_ = "Exploratory data\n" + ("plane" if y[n] else "not plane")
#     plt.imshow(img_)
#     plt.title(label_)
#     plt.show()
## --------------------------------------------

## convert to grayscale and flatten data
def gryImages(images):
    gry_images = []
    for x in images:
        img = cv.imread(x)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gry_images.append(img)
    return gry_images

X = np.array(gryImages(X))

## --------------------------------------------

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
input_shape = (X.shape[1], X.shape[2], 1)
## --------------------------------------------

## train test split
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=rand_state)
## --------------------------------------------
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
## normalize
X_train /= 255
X_test /= 255
print()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
##------------------------------------------------

batch_size = 128
epochs = 12

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

## -------------------------------------------------------------------
# save model
model_name = "./models/lenet_like_model.h5"
model.save(model_name)
