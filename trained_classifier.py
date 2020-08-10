#image classifier testing
#dito15
import pickle
from skimage.io import imread_collection, imshow, imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io
imgtest = imread('images/small/a-pos-0.jpg')
imgtest = rgb2gray(imgtest)
imgtest = resize(imgtest,(77,65),mode='constant', anti_aliasing=False)
imgtest = imgtest.flatten('C')
imgtest = imgtest.reshape(1,-1)
with open('10aug2020p5n5e10_100a0-01-SGD.pkl', 'rb') as file:
    model = pickle.load(file)
#class prediction
prob = model.predict(imgtest)
if prob >= 0.5:
    class_predicted = 1
elif prob < 0.5:
    class_predicted = 0
print("probability: ", prob)
print("Class predicted: ", class_predicted)
#probability
#prob =  model.predict_proba(imgtest)[:,1]
#prob = float(prob)
#print("{0:.2f}".format(prob))