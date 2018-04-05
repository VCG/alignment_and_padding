import os
import cv2
import h5py
import numpy as np

def compute_mask(imgs, n_imgs, nTransforms):
	
	#with h5py.File("output/transforms.h5", "r") as fd:
    #	nTransforms = fd["nTransforms"][:]
	#print(nTransforms.shape)
	masks = []
	for img, n_img, transform in zip(imgs, n_imgs, nTransforms):
		nH, nW = n_img.shape
		tmp = np.ones_like(img)
		img_transformed = cv2.warpAffine(tmp, transform[:2,:], (nW, nH), flags=cv2.INTER_NEAREST)
		masks.append(img_transformed) 

	return np.asarray(masks)

