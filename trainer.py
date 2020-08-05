from dbn import DBN
from skimage.color import rgb2gray
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import numpy as np
model = DBN(n_nodes=5005,rbm_epoch=100,max_epoch=1000, alpha=0.01)
imgs = imread_collection('images/train/*.jpg')
print("Imported", len(imgs), "images")
print("The first one is",len(imgs[0]), "pixels tall, and",
     len(imgs[0][0]), "pixels wide")
imgs = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgs]
imgs = [rgb2gray(x) for x in imgs]
imgsarr = [x.flatten('C') for x in imgs]
#print(np.array(imgsarr))
'''
X = np.array([[0.2157, 0.1255, 0.4039, 1.0, 0.0941, 0.2550],
                [0.1686, 0.9529, 0.0824, 0.0980, 1.0, 0.3529],
                [0.3529, 0.0824, 0.4275, 1.0, 0.1255, 0.2941],
                [0.1255, 1.0, 0.1216, 0.0471, 1.0, 0.2431]])
'''

y = np.array([[0,1,0,1]])
model.fit(np.array(imgsarr), y)
print(model.predict(np.array(imgsarr[1])))