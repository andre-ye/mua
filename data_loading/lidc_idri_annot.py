'''
Arranges the highest-disagreement images from the LIDC-IDRI dataset
and arranges them into the associated directory.

For use in image annotations.
For model training, see lidc_idri_training.py.

A subset of the LIDC-IDRI dataset used for this script is published
on Kaggle at the following link:
https://www.kaggle.com/datasets/washingtongold/lidcidri30

Make sure pylidc is installed.
'''

# CONFIG
AGREEMENT_LEVEL = 0.4
NUM_ANNOTATORS = 6

# IMPORTS
import numpy as np, cv2, os, pandas as pd, tqdm as tqdm, matplotlib.pyplot as plt
import pylidc as pl
from pylidc.utils import consensus
from skimage.measure import find_contours
plt.set_cmap('gray')

# SETUP
if not os.path.exists('imgs'):
  os.mkdir('imgs')
logs = {'Image Name':[],
        'Annotator':[],
        'PID':[],
        'Cluster':[]}

# MAIN LOOP
for CURRINDEX in range(3):
    
    paths = [(range(1, 201), '../input/lidcidri30/LIDC-IDRI-0001-0200'),
         (range(201, 401), '../input/lidcidri30/LIDC-IDRI-0201-0400'),
         (range(401, 601), '../input/lidcidri30/LIDC-IDRI-0401-0600')]

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
        print(pid)

        try:

            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            vol = scan.to_volume()

            nods = scan.cluster_annotations()

            for i_, anns in enumerate(nods):

                try:

                    cmask,cbbox,masks = consensus(anns, clevel=0.5,
                                                  pad=[(20,20), (20,20), (0,0)])

                    k = int(0.5*(cbbox[2].stop - cbbox[2].start))

                    imgs = [find_contours(masks[j][:,:,k])[0] for j in range(len(masks)) if find_contours(masks[j][:,:,k]) != []]
                    annots = []
                    for img in imgs:
                        canvas = np.zeros(vol[cbbox][:,:,k].shape,dtype=np.uint8)
                        cv2.fillPoly(canvas, [np.round(img).astype(np.int32)], 1)
                        annots.append(canvas)       
                    agreement = iou(annots)

                    if agreement < AGREEMENT_LEVEL:
                        
                        for annotator in range(NUM_ANNOTATORS):
                            filename = f'imgs/{annotator}_img-LIDC-IDRI-{code}_cluster-{i_}.png'
                            logs['Image Name'].append(filename)
                            logs['Annotator'].append(annotator)
                            logs['PID'].append(pid)
                            logs['Cluster'].append(i_)
                            plt.imsave(filename, vol[cbbox][:,:,k])
                            zipped += 1

#                         anns = nods[0]
#                         cmask,cbbox,masks = consensus(anns, clevel=0.5,
#                                                       pad=[(20,20), (20,20), (0,0)])
#                         k = int(0.5*(cbbox[2].stop - cbbox[2].start))

#                         fig, ax = plt.subplots(1,1,figsize=(15,5), dpi=400)
#                         plt.subplot(1, 3, 2)
#                         plt.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)
#                         plt.axis('off')
#                         plt.subplot(1, 3, 3)
#                         plt.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)
#                         colors = ['r', 'g', 'b', 'y']
#                         for j in range(len(masks)):
#                             for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
#                                 label = "Annotation %d" % (j+1)
#                                 plt.plot(c[:,1], c[:,0], colors[j], label=label)
#                         for c in find_contours(cmask[:,:,k].astype(float), 0.5):
#                             plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')
#                         plt.axis('off')
#                         plt.legend()
#                         plt.subplot(1, 3, 1)
#                         plt.axis('off')
#                         cmask,cbbox,masks = consensus(anns, clevel=0.5,
#                                                   pad=[(40,40), (30,30), (0,0)])
#                         k = int(0.5*(cbbox[2].stop - cbbox[2].start))
#                         plt.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)
#                         plt.show()

                except:

                    pass

        except:

            pass
