from dbn import DBN
from skimage.color import rgb2gray
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import numpy as np
import pickle
model = DBN(n_nodes=5005,rbm_epoch=120,max_epoch=5000, alpha=0.01, threshold=0.011)

imgs = imread_collection('images/small/*.jpg')
imgstest = imread_collection('images/test/*.jpg')
print("Imported", len(imgs), "images")
print("The first one is",len(imgs[0]), "pixels tall, and",
     len(imgs[0][0]), "pixels wide")
imgs = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgs]
imgs = [rgb2gray(x) for x in imgs]
imgsarr = [x.flatten('C') for x in imgs]

imgstest = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgstest]
imgstest = [rgb2gray(x) for x in imgstest]
imgsarrtest = [x.flatten('C') for x in imgstest]
print(np.array(imgsarr).shape)
'''
X = np.array([[0.2157, 0.1255, 0.4039, 1.0, 0.0941, 0.2550],
                [0.1686, 0.9529, 0.0824, 0.0980, 1.0, 0.3529],
                [0.3529, 0.0824, 0.4275, 1.0, 0.1255, 0.2941],
                [0.1255, 1.0, 0.1216, 0.0471, 1.0, 0.2431]])

x_test = np.array([[0.2157, 0.1255, 0.4039, 1.0, 0.0876, 0.2550],
                [0.1886, 0.9529, 0.0824, 0.0980, 1.0, 0.3529]])
'''
y = []
for i in range(5):
     y.append(1)
for i in range(5):
     y.append(0)
y = np.array([y])

y_test = []
for i in range(13):
     y_test.append(1)
for i in range(13):
     y_test.append(0)
y_test = np.array([y_test])
'''
y = np.array([[0,1,0,1]])
y_test = np.array([[0,1]])
model.fit(X, y, x_test, y_test)
'''
model.fit(np.array(imgsarr), y, np.array(imgsarrtest), y_test)
print(model.predict(np.array([imgsarr[1]])))
filename = '21aug2020p5n5e120_5000t0-011a0-01-SGD.pkl'
pickle.dump(model, open(filename, 'wb'))
