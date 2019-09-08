import csv
from training import constants
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense


def read_training_data():
    center_images = []
    steering_angles = []

    with open('./training/driving_log.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            center_images.append(row[constants.IMAGE_RIGHT])
            steering_angles.append(float(row[constants.STEERING_ANGLE]))

    return np.array(center_images), np.array(steering_angles)

def train(X_train, y_train):
    a = Input(shape=(43,))
    b = Dense(32)(a)
    model = Model(inputs=a, outputs=b)


X_train, y_train = read_training_data()






