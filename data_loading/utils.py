import numpy as np

def iou(imgs):
  intersection = np.logical_and.reduce(imgs)
  union = np.logical_or.reduce(imgs)
  iou_score = np.sum(intersection) / np.sum(union)
  return np.clip(0, 1, iou_score)
