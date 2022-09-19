'''
Trains a segmentation model on standard and derived min/max
annotations. See data_loading/lidc_idri_training.py for the
data source. Compares results and saves weights.
'''

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
plt.set_cmap('gray')

# seeding
random_state = 42
os.environ['PYTHONHASHSEED'] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)
ia.seed(random_state)

# define utilities
from keras import backend as K
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.0001)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# create models
from model_attn_unet import custom_unet
standard_model = custom_unet((100, 100, 1),
                             scaled_size = (128, 128),
                             num_classes = 1,
                             activation = 'relu',
                             use_batch_norm = True,
                             upsample_mode = 'deconv',
                             dropout = 0.1,
                             dropout_change_per_layer = 0.0,
                             dropout_type = 'spatial',
                             use_attention = True,
                             filters = 8,
                             num_layers = 4,
                             output_activation='sigmoid')
minmax_model = custom_unet((100, 100, 1),
                           scaled_size = (128, 128),
                           num_classes = 2,
                           activation = 'relu',
                           use_batch_norm = True,
                           upsample_mode = 'deconv',
                           dropout = 0.1,
                           dropout_change_per_layer = 0.0,
                           dropout_type = 'spatial',
                           use_attention = True,
                           filters = 8,
                           num_layers = 4,
                           output_activation='sigmoid')

# create datasets
from dataset_struct import Dataset
standard_data = Dataset(img_path = '../input/joint-disagreement-lidc-idri-approach/images/images',
                        mask_path = '../input/joint-disagreement-lidc-idri-approach/images/standard_maps',
                        model = standard_model,
                        valid_loss = dice_coef_loss,
                        train_split = 0.8,
                        train_split_seed = random_state,
                        batch_size = 16,
                        apply_augment = True,
                        valid_visualize = True)
minmax_data = Dataset(img_path = '../input/joint-disagreement-lidc-idri-approach/images/images',
                      mask_path = '../input/joint-disagreement-lidc-idri-approach/images/minmax_maps',
                      model = minmax_model,
                      valid_loss = dice_coef_loss,
                      train_split = 0.8,
                      train_split_seed = random_state,
                      batch_size = 16,
                      apply_augment = True,
                      valid_visualize = True)

# model training
standard_model.compile(optimizer = 'adam',
                       loss = dice_coef_loss,
                       metrics = ['mse', 'mae'])
shistory = standard_model.fit(standard_data, epochs = 600)
shistory.save_weights('standard_model.h5')

minmax_model.compile(optimizer = 'adam',
                     loss = dice_coef_loss,
                     metrics = ['mse', 'mae'])
mhistory = minmax_model.fit(minmax_data, epochs = 600)
minmax_model.save_weights('minmax_model.h5')

# save history
plt.figure(figsize=(10, 5))
plt.plot(shistory.history['loss'], label='Standard', color='red')
plt.plot(mhistory.history['loss'], label='Min Max', color='blue')
plt.title('Training Loss History')
plt.legend()
plt.show()
plt.savefig('train_loss.png', dpi=400)

plt.figure(figsize=(10, 5))
plt.plot(standard_data.valid_history, label='Standard', color='red')
plt.plot(minmax_data.valid_history, label='Min Max', color='blue')
plt.title('Validation Loss History')
plt.legend()
plt.show()
plt.savefig('valid_loss.png', dpi=400)
