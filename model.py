import csv
from training import constants
import numpy as np
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, Lambda, Cropping2D, MaxPool2D, Dropout
from keras.callbacks import EarlyStopping


def read_training_data(file, mirror_input=True, multicam=True):
    images = []
    steering_angles = []

    with open(file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            row_images = []
            row_steering_angles = []
            # create adjusted steering measurements for the side camera images
            steer_correction = 0.2  # this is a parameter to tune

            row_images.append(cv2.imread(row[constants.IMAGE_CENTER]))
            row_steering_angles.append(float(row[constants.STEERING_ANGLE]))

            if multicam:
                row_images.append(cv2.imread(row[constants.IMAGE_LEFT]))
                row_images.append(cv2.imread(row[constants.IMAGE_RIGHT]))

                row_steering_angles.append(
                    float(row[constants.STEERING_ANGLE]) + steer_correction)
                row_steering_angles.append(
                    float(row[constants.STEERING_ANGLE]) - steer_correction)

            if mirror_input:
                mirrored_row_images, mirrored_row_steering_angles = mirror(
                    row_images, row_steering_angles)
                row_images.extend(mirrored_row_images)
                row_steering_angles.extend(mirrored_row_steering_angles)

            images.extend(row_images)
            steering_angles.extend(row_steering_angles)

    return np.array(images), np.array(steering_angles)


def mirror(images, steering_angles):
    return list(map(lambda image: cv2.flip(image, 1), images)), list(map(lambda angle: -angle, steering_angles))


def crop(image, top=64, bottom=20, left=0, right=0):
    height = image.shape[0]
    width = image.shape[1]
    return image[top:(height - bottom), left:(width - right)]


def train(X_train, y_train, keep_prob=0.8):
    dropout_rate = 1.0 - keep_prob

    model = Sequential()
    model.add(Cropping2D(((40, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x - 128) / 255))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    # model.add(MaxPool2D((2, 2), (1, 1)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    # model.add(MaxPool2D((2, 2), (1, 1)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    # model.add(MaxPool2D((2, 2), (1, 1)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPool2D((2, 2), (1, 1)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPool2D((2, 2), (1, 1)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(50))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(10))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train, validation_split=0.2, epochs=20,
              shuffle=True, callbacks=[EarlyStopping(patience=8)])

    return model


X_train, y_train = read_training_data('./training/driving_log.csv')

model = train(X_train, y_train)

model.save('model.h5')
