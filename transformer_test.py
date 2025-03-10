import sys
if './vision_transformer' not in sys.path:
    sys.path.append('./vision_transformer')

dataset_path = 'nabirds-data/nabirds/'
image_path = dataset_path + 'images/'

import flax
import jax
import numpy as np
import optax
import tqdm

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import utils
from vit_jax import models
from vit_jax import train
from vit_jax.configs import augreg as augreg_config
from vit_jax.configs import models as models_config
from vit_jax.configs import common as common_config
import ml_collections

import utils

import glob
import os
import random
import shutil
import time

from absl import logging
import pandas as pd
import seaborn as sns
import tensorflow as tf

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

pd.options.display.max_colwidth = None
logging.set_verbosity(logging.INFO) # Show logs in training

config = common_config
config.batch = 32

ds_train, ds_test = utils.tf_load_images(image_path, batch_size = 32, img_size=(224,224))
num_classes = input_pipeline.get_directory_info(image_path)


#print(ds_train)
































