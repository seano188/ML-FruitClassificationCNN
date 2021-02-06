import numpy as np
import pandas as pd
import PIL.Image
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Import the training image dataset, and append the images and labels to an array
x_train = []
y_train = []

train_loc = 'data/train'

for filename_train in os.listdir(train_loc):
    if filename_train.split('.')[1] == 'jpg':
        img_train = cv2.imread(os.path.join(train_loc, filename_train))
        img_ary_train = PIL.Image.fromarray(img_train, 'RGB')
        img_resize_train = img_ary_train.resize((64, 64))
        img_train_arr = np.array(img_resize_train)
        rgb_img_train_arr = cv2.cvtColor(img_train_arr, cv2.COLOR_BGR2RGB)
        x_train.append(rgb_img_train_arr)
        y_train.append(filename_train.split('_')[0])


# Import the test image dataset, and append the images and labels to an array
x_test = []
y_test = []

test_loc = 'data/test'

for filename_test in os.listdir(test_loc):
    if filename_test.split('.')[1] == 'jpg':
        img_test = cv2.imread(os.path.join(test_loc, filename_test))
        img_ary_test = PIL.Image.fromarray(img_test, 'RGB')
        img_resize_test = img_ary_test.resize((64, 64))
        img_test_arr = np.array(img_resize_test)
        rgb_img_test_arr = cv2.cvtColor(img_test_arr, cv2.COLOR_BGR2RGB)
        x_test.append(rgb_img_test_arr)
        y_test.append(filename_test.split('_')[0])

# To verify the number of labels and images
print("Unique labels for train: ", np.unique(y_train))
print("Number of jpg images in train are: ", len(x_train))

print("\nUnique labels for test: ", np.unique(y_test))
print("Number of jpg images for test are: ", len(x_test))

# To check the first image
img_number = random.randint(0,100)
image = x_train[img_number]

plt.imshow(image)
plt.title("Label: " + y_train[img_number])
plt.show()

# To view data for x_train
print(x_train)

x_train = np.array(x_train).astype('float32')
x_test = np.array(x_test).astype('float32')

x_test /= 255
x_train /= 255

# To one-hot encode the single digit labels of y_train and y_test and to convert to 4 dimensional array
le = LabelEncoder()
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_train_encode = le.fit_transform(y_train[0])
y_test_encode = le.fit_transform(y_test[0])

y_train = tf.keras.utils.to_categorical(y_train_encode, 4)
y_test = tf.keras.utils.to_categorical(y_test_encode, 4)

# To view data for y_train after one-hot encoding
print(y_train)

# Splitting of training dataset into training and validation sets with a 70:30 split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

#Building of model
model = tf.keras.Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.0001), padding="same",
                 input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.0001), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.0001), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(4, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.0003)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Populate training dataset with more images using an image generator
datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    width_shift_range=.15,
    height_shift_range=.15,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)
batchsize = 40

# Prepare training model for evaluation against the validation data
history = model.fit(datagen.flow(x_train, y_train, batch_size=batchsize),
                    epochs=50, steps_per_epoch=x_train.shape[0] / batchsize, validation_data=(x_valid, y_valid), verbose=1)

# Run test data against training model and print accuracy/loss values
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Generate a plot of the accuracy and loss chart for the training data and validation data
figure = plt.figure(figsize=(15, 15))
ax = figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.title.set_text('model accuracy')
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend(['Training Accuracy', 'Val Accuracy'])
bx = figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.title.set_text('model loss')
bx.set_ylabel('loss')
bx.set_xlabel('epoch')
bx.legend(['Training Loss', 'Val Loss'])
plt.show()
