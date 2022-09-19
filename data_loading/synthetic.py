'''
Generates synthetic data. More information coming.
'''

import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.set_cmap('gray')

DIM = 500
NUM_IMAGES = 40
NUM_ANNOTATORS = 5

def place_circle(canvas,
                 shape_min = 40,
                 shape_max = 70,
                 blur_min = 1,
                 blur_max = 60,
                 color_min = 0,
                 color_max = 0.075,
                 min_x = 0,
                 min_y = 0,
                 lower_rad_prop = 0.05,
                 upper_rad_prop = 0.2,
                 x = None,
                 y = None
                 ):
    radius = int(np.random.uniform(int(DIM * lower_rad_prop), int(DIM * upper_rad_prop)))
    if not x and not y:
        x = int(np.random.uniform(min_x + radius, DIM - radius - min_x))
        y = int(np.random.uniform(min_y + radius, DIM - radius - min_y))
    canvas = cv2.circle(canvas, (x, y), radius,
                        np.random.uniform(color_min, color_max),
                        thickness=-1)
    canvas = cv2.blur(canvas, (np.random.randint(blur_min, blur_max+1), np.random.randint(blur_min, blur_max+1)))
    return canvas
  
def birth_image():
    canvas = np.zeros((DIM,DIM))
    for i in range(6):
        canvas = place_circle(canvas,
                              min_x = np.floor(DIM / 3),
                              min_y = np.floor(DIM / 3),
                              color_min = 0.95,
                              color_max = 1,
                              blur_min = 10,
                              blur_max = 15,
                              lower_rad_prop = 0.05,
                              upper_rad_prop = 0.1)

    for i in range(20):
        place_circle(canvas,
                     min_x=100,
                     min_y=100,
                     color_min = 0.9,
                     color_max = 1,
                     blur_min = 1,
                     blur_max = 5,
                     lower_rad_prop = 0.01,
                     upper_rad_prop = 0.05)
    
    for i in range(40):
        place_circle(canvas,
                     color_min = 0.9,
                     color_max = 1,
                     blur_min = 1,
                     blur_max = 1,
                     lower_rad_prop = 0.0075,
                     upper_rad_prop = 0.04)
    
    for i in range(2):
        canvas = cv2.blur(canvas, (15, 15))

    return canvas
 
os.makedirs('images')
images = [birth_image() for i in range(NUM_IMAGES)]
for annotator in range(NUM_ANNOTATORS):
    for index, image in enumerate(images):
        plt.imsave(f'images/{annotator}_standard_{index}.png', image)
    for index, image in enumerate(images):
        plt.imsave(f'images/{annotator}_minmax_{index}.png', image)
shutil.make_archive('images', 'zip', 'images')
