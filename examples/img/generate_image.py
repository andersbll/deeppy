from skimage.io import imshow
from skimage import io
import numpy as np
import skimage
from skimage.transform import rotate
import random
import cPickle as pickle
import glob
import os

X = []
Y = []
bi = 0
brickDir = "/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/org_img/brick/"
carDir = "/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/org_img/carp/"
for brickFile, carpFile in zip(os.listdir(brickDir), os.listdir(carDir)):

    bi += 1
    img_wood = io.imread(brickDir + brickFile)
    img_brick = io.imread(carDir + carpFile)
    
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


    img_mix3 = img_brick[0:imageSize,0:imageSize]
    y3 = np.empty(img_mix2.shape)

    for pi in range(imageSize):
        img_mix3[0:pi+1,pi:imageSize] = img_wood[0:pi+1,pi:imageSize]
        #img_mix3[pi:imageSize,0:pi] = img_brick[pi:imageSize,0:pi]

        y3[0:,pi:imageSize] = 255
        y3[pi:imageSize,0:pi] = 0
    
    
    for d in range(4):
        img1 = rotate(img_mix1, 90*d)
        img2 = rotate(img_mix2, 90*d)
        img3 = rotate(img_mix3, 90*d)
    
        X.append(img1)
        X.append(img2)
        X.append(img3)
    
        true1 = rotate(y1, 90*d)
        true2 = rotate(y2, 90*d)
        true3 = rotate(y3, 90*d)
    
        Y.append(true1)
        Y.append(true2)
        Y.append(true3)


combined = zip(X, Y)
random.shuffle(combined)

X[:], Y[:] = zip(*combined)

pickle.dump(X, open( '/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/X.pic', "wb" ))
pickle.dump(Y, open( '/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/Y.pic', "wb" ))


#'/Users/lasse/Documents/DTU/Master/RemoteCode/deeppy/examples/img/pattern