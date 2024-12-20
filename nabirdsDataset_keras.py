import os
import sys
import keras

dataset_path = 'nabirds-data/nabirds/'
image_path = dataset_path + 'images/'


def load_images(dir=image_path, batch_size=32, img_size=(256, 256), seed=0, split=0.8):
    training, validation = keras.utils.image_dataset_from_directory(
                directory = dir,
                labels = "inferred",
                label_mode = "categorical",
                class_names = None,
                color_mode = "rgb",
                batch_size = batch_size,
                image_size = img_size,
                shuffle = True, 
                seed = seed,
                validation_split = split,
                subset = "both",
                interpolation = "bilinear",
                follow_links = False,
                crop_to_aspect_ratio = False,
                pad_to_aspect_ratio = False,
                data_format = None,
                verbose = True,
            )

    return training, validation


if __name__ == "__main__":
    load_images()
