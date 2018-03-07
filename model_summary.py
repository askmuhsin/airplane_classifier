from keras.models import Sequential

model = Sequential()
model_location = './ann_dense1.h5'
model.load_weights(model_location)
print("Loaded model from disk:")
model.summary()
