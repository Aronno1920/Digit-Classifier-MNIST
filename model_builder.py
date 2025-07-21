###### Import Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Best epochs number using EarlyStopping & ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#################


################## Declare method
def train_model():

    ###### Load dataset
    data = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data
    #################


    ###### Normalization Processing
    # Normalize pixel values (0-255 → 0-1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten the 28x28 images to 784-dimensional vectors
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    #################


    ###### Model Architecture
    model = Sequential([
        Input(shape=(784,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.2),

        Dense(10, activation='softmax')
    ])

    model.summary()
    #################


    ###### Model Training - Use EarlyStopping
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath='model/mnist_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stop, model_checkpoint],
        verbose=2
    )
    #################


################## Return value
    return history.history
#################
