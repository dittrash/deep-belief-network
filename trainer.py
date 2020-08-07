from dbn import DBN
from skimage.color import rgb2gray
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import numpy as np
import pickle

#build model
model = DBN(n_nodes=5005,rbm_epoch=100,max_epoch=2000, alpha=0.001)

#import training data
imgs_train = imread_collection('images/train/*.jpg')
print("Imported", len(imgs_train), "images")
print("The first one is",len(imgs_train[0]), "pixels tall, and",
     len(imgs_train[0][0]), "pixels wide")
imgs_train = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgs_train]
imgs_train = [rgb2gray(x) for x in imgs_train]
imgsarr_train = [x.flatten('C') for x in imgs_train]
print(np.array(imgsarr_train).shape)
'''
X = np.array([[0.2157, 0.1255, 0.4039, 1.0, 0.0941, 0.2550],
                [0.1686, 0.9529, 0.0824, 0.0980, 1.0, 0.3529],
                [0.3529, 0.0824, 0.4275, 1.0, 0.1255, 0.2941],
                [0.1255, 1.0, 0.1216, 0.0471, 1.0, 0.2431]])
'''
y = []
for i in range(250):
     y.append(1)
for i in range(250):
     y.append(0)
y = np.array([y])

model.fit(np.array(imgsarr_train), y)
model.predict(np.array([imgsarr_train[0]]))
filename = '8aug2020p250n250e100_2000a0-001_0-01_0-1.pkl'
pickle.dump(model, open(filename, 'wb'))
