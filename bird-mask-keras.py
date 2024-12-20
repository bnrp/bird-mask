import nabirds
from nabirdsDataset import nabirdsDataset
from utils import load_train_test, timer, load_data
from nabirdsDataset_keras import load_images, load_images_with_hierarchy

import time
import os

# os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras.applications import ResNet50V2
import numpy as np
import tensorflow as tf
#import torch


def init_model(include_top=False, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, activation="softmax", name="resnet50v2"):
    model = ResNet50V2(
                include_top=include_top, 
                weights=weights, 
                input_tensor=input_tensor,
                input_shape=input_shape,
                pooling=pooling,
                classes=classes,
                classifier_activation=activation,
                name=name,
            )
    
    return model







def main():
    # Model Variables
    batch_size = 16
    img_size = (256, 256)
    input_size = (256, 256, 3)
    seed = 0
    split = 0.8
    lr = 1e-3
    epochs = 100


    # Dataset locations
    dataset_path = 'nabirds-data/nabirds/'
    image_path = dataset_path + 'images/'


    # Load data
    training_set, validation_set = load_images_with_hierarchy(
                dir=image_path,
                batch_size=batch_size,
                img_size=img_size,
                seed=seed,
                split=split,
            )

    #print(training_set)

    num_classes = training_set.element_spec[1].shape[1]


    # Create model
    res_model = init_model(
                weights=None,
                input_shape=input_size,
                classes=num_classes,
            )

    #for layer in res_model.layers[:143]:
    #    layer.trainable = False


    model = keras.models.Sequential()
    model.add(res_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes*8, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes*4, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes*2, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

   
    tf.autograph.set_verbosity(
        level=0, alsologtostdout=False
    )

    # Train model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(training_set, epochs=epochs, verbose=1, validation_data=validation_set)



if __name__ == '__main__':
    main()
