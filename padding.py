import os
import cv2
import numpy as np

def compute_padding(imgs, transforms):
	for i in range(len(transforms)):
		#compute transofrms for bnd pts
		(h,w) = imgs[i].shape
		bnd_pts_src = np.array([[0,0,1], [w,0,1], [0,h,1], [w,h,1]])
		if i == 0:	
			bnd_pts_dest = np.dot(transforms[i], bnd_pts_src.T).T
		else:
			bnd_pts_dest = np.append( bnd_pts_dest, np.dot(transforms[i], bnd_pts_src.T).T, axis=0 )

	ymin = np.floor(min(bnd_pts_dest[:, 1]))
	ymax = np.ceil(max(bnd_pts_dest[:, 1]))
	xmin = np.floor(min(bnd_pts_dest[:, 0]))
	xmax = np.ceil(max(bnd_pts_dest[:, 0]))
		
	#print(xmin,ymin,xmax,ymax)

	nH = int(ymax - ymin)
	nW = int(xmax - xmin)
	nTx = -xmin
	nTy = -ymin

	n_imgs = []
	nTransforms = []

	for img, transform in zip(imgs, transforms):
		nTransform = transform.copy()
		nTransform[0, 2] = nTransform[0, 2] + nTx
		nTransform[1, 2] = nTransform[1, 2] + nTy
		img_transformed = cv2.warpAffine(img, nTransform[:2,:], (nW, nH), flags=cv2.INTER_AREA)
		n_imgs.append(img_transformed) 
		nTransforms.append(nTransform)

	return np.asarray(n_imgs), np.asarray(nTransforms)


