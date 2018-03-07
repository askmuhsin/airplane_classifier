import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2 as cv
import random
import numpy as np
from sklearn.svm import LinearSVC
import time
import pickle

# --------------------------------------------

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
# --------------------------------------------

X = not_plane + plane
y = label

assert len(X)==len(y), "mismatch in data and label size"

# # Exploratory data
# for _ in range(5):
#     n = random.randint(0, len(X)-1)
#     img_ = plt.imread(X[n])
#     label_ = "plane" if y[n] else "not plane"
#     plt.imshow(img_)
#     plt.title(label_)
#     plt.show()
# --------------------------------------------

# convert to grayscale and flatten data
def gryAndFlatten(images):
    flatten_data =[]
    for x in images:
        img = cv.imread(x)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flatten_data.append(img.flatten())
    return flatten_data

X = np.array(gryAndFlatten(X))

assert X.shape[0]==len(X)
# --------------------------------------------

# train test split
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=rand_state)
# --------------------------------------------

## implement support vector classifier

t = time.time()
svc = LinearSVC()
svc.fit(X_train, y_train)
print("time take to train SVC:\t", time.time()-t)

filename = 'svc_model1.p'
pickle.dump(svc, open(filename, 'wb'))

print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
