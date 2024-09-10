import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import cv2
import os
import random

unpath = "C:/Users/minth/PythonProject/bottle/unlabel/"
path = "C:/Users/minth/PythonProject/bottle/label/"
canpath = "C:/Users/minth/PythonProject/bottle/can/"
train_X, train_Y = [], []
leng = 0

for i in glob(unpath+"*.jpg"):
    leng += 1
    IMG = cv2.imread(i)
    IMG = cv2.resize(IMG, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    train_X.append(IMG)
    train_Y.append(0)
for i in glob(path+"*.jpg"):
    leng += 1
    IMG = cv2.imread(i)
    IMG = cv2.resize(IMG, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    train_X.append(IMG)
    train_Y.append(1)
for i in glob(canpath+"*.jpg"):
    leng += 1
    IMG = cv2.imread(i)
    IMG = cv2.resize(IMG, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    train_X.append(IMG)
    train_Y.append(2)
    
train_X = np.array(train_X)
train_Y = np.array(train_Y)
train_X = train_X / 255.0
t = np.array([ i for i in range(leng) ])
np.random.shuffle(t)
train_X = train_X.reshape(-1, 300, 300, 3)
train_X=train_X[t]
train_Y=train_Y[t]

test_X = train_X[:100]
test_Y = train_Y[:100]
train_X = train_X[100:]
train_Y = train_Y[100:]

print(test_X.shape, test_Y.shape, train_X.shape, train_Y.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(300, 300, 3), kernel_size=(3,3), filters=16),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

checkpoint_path = "C:/Users/minth/PythonProject/models"
checkpoint_dir = os.path.dirname(checkpoint_path)

history = model.fit(train_X, train_Y, epochs=50, batch_size=2, validation_split=0.25)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

model.evaluate(test_X, test_Y, verbose=0)

model.save("C:/Users/minth/PythonProject/BottleModel.h5")
