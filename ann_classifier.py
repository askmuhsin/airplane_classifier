import glob
import time
import pickle
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

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
y = label

assert len(X)==len(y), "mismatch in data and label size"

## --------------------------------------------
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
def gryAndFlatten(images):
    flatten_data =[]
    for x in images:
        img = cv.imread(x)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flatten_data.append(img.flatten())
    return flatten_data

X = np.array(gryAndFlatten(X))

assert X.shape[0]==len(X)
## --------------------------------------------

## train test split
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=rand_state)
## --------------------------------------------

## ANN - model
model = Sequential()

model.add(Lambda(lambda x:x/255.0-0.5, input_shape=X_train[0].shape))
# model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=10)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
