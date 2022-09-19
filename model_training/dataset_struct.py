import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm
import random

class Dataset(tf.keras.utils.Sequence):
    
    def __init__(self,
                 img_path,               # path to images (numpy arrays)
                 mask_path,              # path to masks (numpy arrays)
                 model,                  # reference to model
                 valid_loss,             # validation loss function
                 train_split = 0.8,
                 train_split_seed = 42,
                 batch_size = 64,
                 apply_augment = False,
                 aug_action_prob = 0.5,
                 crop_const = 0.1,
                 scale_const = 0.1,
                 trans_const = 0.1,
                 rot_const = 10,
                 blur_const = 0.2,
                 sharpen_const = 0.2,
                 light_const = 0.2, 
                 contrast_const = 0.2,
                 img_scale_const = 2**12,
                 valid_visualize = True):
        
        # set up data
        self.sample_ids = [name.split('.')[0] for name in os.listdir(img_path)]
        ids_length = len(self.sample_ids)
        np.random.seed(train_split_seed)
        self.train_idxs = np.random.choice(np.arange(ids_length),
                                           replace = False,
                                           size = int(np.floor(train_split * ids_length)))
        self.valid_idxs = np.array([idx for idx in np.arange(ids_length) \
                                    if idx not in self.train_idxs])
        self.num_train_batches = len(self.train_idxs) // batch_size
        self.num_valid_batches = len(self.valid_idxs) // batch_size
        self.num_total_batches = ids_length // batch_size
        
        # initialize history recording
        self.valid_history = []
        
        # set up data pipeline
        self.aug_pipe = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.GaussianBlur((0, blur_const)),
                iaa.Sharpen(alpha = (0, sharpen_const), 
                            lightness = (1-light_const, 1+light_const)),
                iaa.LinearContrast((1-contrast_const, 1+contrast_const))
            ],
            random_order = True
        )
        
        sometimes = lambda aug: iaa.Sometimes(aug_action_prob, aug)
        self.aug_pipe = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Crop(percent=(0, crop_const))),
                sometimes(iaa.Affine(
                    scale={"x": (1-scale_const, 1+scale_const), "y": (1-scale_const, 1+scale_const)},
                    translate_percent={"x": (-trans_const, trans_const), "y": (-trans_const, trans_const)},
                    rotate=(-rot_const, rot_const),
                    order=[0, 1],
                    mode=ia.ALL
                )),
                sometimes(iaa.GaussianBlur((0, blur_const))),
                sometimes(iaa.Sharpen(alpha=(0, sharpen_const), lightness=(1-light_const, 1+light_const))),
                sometimes(iaa.LinearContrast((1-contrast_const, 1+contrast_const)))
            ],
            random_order = True
        )
        
        # save relevant variables
        self.img_path, self.mask_path = img_path, mask_path
        self.apply_augment = apply_augment
        self.batch_size = batch_size
        self.model = model
        self.valid_loss = valid_loss
        self.divide_const = img_scale_const
        self.valid_visualize = valid_visualize
        
    def augment(self, imgs, masks):
        return self.aug_pipe(image=imgs, segmentation_maps=masks)
        
    def __len__(self):
        return self.num_train_batches
    
    def get_imgs(self, start_idx, num_samples, idxs_list, apply_augment):
        for idx in range(start_idx, start_idx + num_samples):
            file_name = self.sample_ids[idxs_list[idx]]
            img = np.load(os.path.join(self.img_path, f'{file_name}.npz'))['arr_0'].astype(np.float32) / self.divide_const
            mask = np.load(os.path.join(self.mask_path, f'{file_name}.npz'))['arr_0'].astype(np.float32)
            img = img.astype(np.float32) / self.divide_const
            if apply_augment:
                img, mask = self.aug_pipe(image=img, segmentation_maps=np.expand_dims(mask, 0).astype(np.uint8))
                mask = np.squeeze(mask, axis=0)
            mask = mask.astype(np.float32)
            yield img, mask
    
    def __getitem__(self, idx):
        
        start_idx = idx * self.batch_size
        imgs, masks = [], []
        for img, mask in self.get_imgs(start_idx, self.batch_size, self.train_idxs, self.apply_augment):
            imgs.append(img)
            masks.append(mask)
        return np.stack(imgs), np.stack(masks)
                         
    def on_epoch_end(self):
        self.valid_history.append(self.validate(visualize=self.valid_visualize))
        
    def validate(self, visualize=False, print_result=True, k=3):
        
        losses = []
        
        for batch_idx in tqdm(range(self.num_valid_batches)):
            
            start_idx = batch_idx * self.batch_size
            imgs, masks = [], []
            for img, mask in self.get_imgs(start_idx, self.batch_size, self.valid_idxs, False):
                imgs.append(img)
                masks.append(mask)
            imgs, masks = np.stack(imgs), np.stack(masks)
            pred = self.model.predict_on_batch(imgs)
            losses.append(self.valid_loss(pred, masks))

            if visualize and k != 0:
                
                if pred.shape[-1] == 1:

                    plt.figure(figsize=(10, 3))
                    plt.subplot(1, 3, 1)
                    plt.imshow(imgs[0])
                    plt.axis('off')
                    plt.title('Image')
                    plt.subplot(1, 3, 2)
                    plt.imshow(pred[0])
                    plt.axis('off')
                    plt.title('Predicted Mask')
                    plt.subplot(1, 3, 3)
                    plt.imshow(masks[0])
                    plt.axis('off')
                    plt.title('Ground Truth')
                    plt.show()
                    
                else:
                    
                    plt.figure(figsize=(15, 3))
                    plt.subplot(1, 5, 1)
                    plt.imshow(imgs[0])
                    plt.axis('off')
                    plt.title('Image')
                    plt.subplot(1, 5, 2)
                    plt.imshow(pred[0,:,:,0])
                    plt.axis('off')
                    plt.title('Predicted Min Mask')
                    plt.subplot(1, 5, 3)
                    plt.imshow(masks[0,:,:,0])
                    plt.axis('off')
                    plt.title('True Min Mask')
                    plt.subplot(1, 5, 4)
                    plt.imshow(pred[0,:,:,1])
                    plt.axis('off')
                    plt.title('Predicted Max Mask')
                    plt.subplot(1, 5, 5)
                    plt.imshow(masks[0,:,:,1])
                    plt.axis('off')
                    plt.title('True Max Mask')
                    plt.show()
                
                k -= 1
                
        if print_result:
            print(f'Validation Loss: {sum(losses) / len(losses)}')
        
        return sum(losses) / len(losses)
