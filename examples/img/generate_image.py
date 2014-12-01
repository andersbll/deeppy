from skimage.io import imshow
from skimage import io
import numpy as np
from skimage.transform import rotate
import random
import cPickle as pickle

X = []
Y = []

for i in range(1, 40):
	img_wood = io.imread('/Users/lasse/Downloads/T11-T15/T14_brick1/T14_%02d.jpg' % i)
	img_brick = io.imread('/Users/lasse/Downloads/T21-T25/T24_corduroy/T24_%02d.jpg' % i)

	imageSize = 128

	ran = random.randrange(-5, 5)
	img_woodn = img_wood[0:imageSize,0:imageSize/2-ran]
	img_brickn = img_brick[0:imageSize,0:imageSize/2+ran]
	
	img_mix1 = np.concatenate([img_woodn, img_brickn], axis=1)
	y1 = np.zeros((imageSize,imageSize), dtype=np.float)
	y1[0:imageSize,0:imageSize/2-ran] = 255
	
	ran = random.randrange(-5, 5)
	img_woodn = img_wood[0:imageSize/2-ran,0:imageSize]
	img_brickn = img_brick[0:imageSize/2+ran,0:imageSize]
	img_mix2 = np.concatenate([img_woodn, img_brickn], axis=0)
	y2 = np.zeros((imageSize,imageSize), dtype=np.float)
	y2[0:imageSize/2-ran,0:imageSize] = 255

	for d in range(4):
		img1 = rotate(img_mix1, 90*d)
		img2 = rotate(img_mix2, 90*d)

		X.append(img1)
		X.append(img2)

		true1 = rotate(y1, 90*d)
		true2 = rotate(y2, 90*d)

		Y.append(true1)
		Y.append(true2)


combined = zip(X, Y)
random.shuffle(combined)

X[:], Y[:] = zip(*combined)

pickle.dump(X, open( '/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/X.pic', "wb" ))
pickle.dump(Y, open( '/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/Y.pic', "wb" ))


#'/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/pattern