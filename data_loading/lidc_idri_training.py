import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from tqdm.notebook import tqdm
plt.set_cmap('gray')

# configuration
SAVE_DIR = 'images'
TOTAL_PADDING = 50
PADDING_MARGIN = 10
SIZE = 100
TEST_PRINT = False
VISUALIZE = False

# create directories
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    os.makedirs(os.path.join(SAVE_DIR, 'images'))
    os.makedirs(os.path.join(SAVE_DIR, 'minmax_maps'))
    os.makedirs(os.path.join(SAVE_DIR, 'standard_maps'))

# define & import relevant utilities
from pylidc.utils import consensus
from skimage.measure import find_contours
def iou(imgs):
        intersection = np.logical_and.reduce(imgs)
        union = np.logical_or.reduce(imgs)
        iou_score = np.sum(intersection) / np.sum(union)
        return np.clip(0, 1, iou_score)

# initialize reading information
num_samples = 0
paths = [(range(1, 201), '../input/lidcidri30/LIDC-IDRI-0001-0200'),
         (range(201, 401), '../input/lidcidri30/LIDC-IDRI-0201-0400'),
         (range(401, 601), '../input/lidcidri30/LIDC-IDRI-0401-0600'),
         (range(600, 801), '../input/lidc-half-2/LIDC_img_0601_0800'),
         (range(801, 1013), '../input/lidc-half-2/LIDC_img_0801_1012')]

# process batches and save
for CURRINDEX in range(len(paths)):

    NUMITER = paths[CURRINDEX][0]
    path = paths[CURRINDEX][1]
    f = open('/root/.pylidcrc', 'w')
    f.write(f'[dicom]\npath = {path}\n\n')
    f.close()

    agrees = []
    zipped = 0

    pad = lambda num, dig: ''.join(['0' for i in range(max(0, dig - len(str(num))))]) + str(num)
    for num in tqdm(NUMITER):
        
        code = pad(num, 4)
        pid = f'LIDC-IDRI-{code}'
        
        try:
          
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            vol = scan.to_volume()
            nods = scan.cluster_annotations()

            for cluster_num, anns in enumerate(nods):
                
                # compute mask pairings and nodes
                cmask,cbbox,masks = consensus(anns, clevel=0.5,
                                              pad=[(TOTAL_PADDING, TOTAL_PADDING), 
                                                   (TOTAL_PADDING, TOTAL_PADDING),
                                                   (0, 0)])

                k = int(0.5*(cbbox[2].stop - cbbox[2].start))
                orig_img = vol[cbbox][:,:,k]
                
                # compute cropping
                center = (int(orig_img.shape[0] / 2), int(orig_img.shape[1] / 2))
                hs = SIZE // 2
                if (center[0]-hs+1 >= 0) and (center[0]-hs+1+SIZE < orig_img.shape[0]) \
                                         and (center[0]-hs+1+SIZE < orig_img.shape[1]):
                    x1 = center[0]-hs+1
                    x2 = center[0]-hs+1+SIZE
                    y1 = center[1]-hs+1
                    y2 = center[1]-hs+1+SIZE
                elif orig.shape[0] >= SIZE and orig.shape[1] >= SIZE:
                    x1, x2 = 0, 0
                    y1, y2 = SIZE, SIZE
                else:
                    continue
                    
                # crop original image
                cropped_patch = orig_img[x1:x2, y1:y2]
                
                # calculate standard mask
                median_annot = find_contours(cmask[:,:,k].astype(float), 0.5)[0]
                canvas = np.zeros(orig_img.shape, dtype=np.uint8)
                coord = [[j, i] for i, j in median_annot]
                coord = np.round(np.array(coord)).astype(np.int32)
                cv2.fillPoly(canvas, [coord], 1)
                cropped_standard = canvas[x1:x2, y1:y2]
                
                # calculate minmax mask
                annots = []
                for annotator in range(len(masks)):
                    canvas = np.zeros(orig_img.shape, dtype=np.uint8)
                    for cluster in find_contours(masks[annotator][:,:,k].astype(float), 0.5):
                        coord = [[j, i] for i, j in cluster]
                        coord = np.round(np.array(coord)).astype(np.int32)
                        cv2.fillPoly(canvas, [coord], 1)
                    annots.append(canvas)

                inter = np.logical_and.reduce(annots)
                union = np.logical_or.reduce(annots)

                cropped_inter = inter[x1:x2, y1:y2]
                cropped_union = union[x1:x2, y1:y2]
                
                # save images and masks
                patch_name = f'{pid}_{cluster_num}'
                np.savez(f"{os.path.join(SAVE_DIR, 'images')}/{patch_name}.npz", np.expand_dims(cropped_patch, -1))
                np.savez(f"{os.path.join(SAVE_DIR, 'minmax_maps')}/{patch_name}.npz", np.stack([cropped_inter, cropped_union], axis=-1))
                np.savez(f"{os.path.join(SAVE_DIR, 'standard_maps')}/{patch_name}.npz", np.expand_dims(cropped_standard, -1))
                
                # sanity checking
                if TEST_PRINT:
                    print(orig_img.shape,
                          cropped_patch.shape,
                          np.stack([cropped_inter, cropped_union], axis=-1).shape,
                          cropped_standard.shape)
                
                if VISUALIZE:
                    plt.figure(figsize=(10, 3), dpi=400)
                    plt.subplot(1, 4, 1)
                    plt.imshow(cropped_patch)
                    plt.title('Image')
                    plt.axis('off')
                    plt.subplot(1, 4, 2)
                    plt.imshow(cropped_inter)
                    plt.title('Intersection (Min)')
                    plt.axis('off')
                    plt.subplot(1, 4, 3)
                    plt.imshow(cropped_union)
                    plt.title('Union (Max)')
                    plt.axis('off')
                    plt.subplot(1, 4, 4)
                    plt.imshow(cropped_standard)
                    plt.title('Standard')
                    plt.axis('off')
                    plt.show()
    
                num_samples += 1
                
        except:

            print(f'{pid} failed :(')
