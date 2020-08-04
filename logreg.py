#impor library yang dibutuhkan
import numpy as np
from sklearn.metrics import accuracy_score
import time

class LogisticReg:
	def __init__(self, n_features, epoch, alpha):
		self.epoch = epoch
		self.alpha = alpha
		self.n_features = n_features
		self.weight = np.ones(self.n_features)

	#fungsi sigmoid
	def sigmoid(self, z):
		return 1/(1+np.exp(-z))
	
    #fungsi loss
	def cost(self, h, y):
		cost = np.sum(y * np.log(h) + (1-y) * np.log(1-h))/-y.shape[0]
		return cost

    #latih
	def fit(self, features, y):
		#menambahkan node bias
		intercept = np.ones((features.shape[0], 1))
		features = np.concatenate((intercept, features), axis=1)
		start_time = time.time()
		for i in range(self.epoch):
			print("LR iteration: ", i)
			z = np.dot(features, self.weight)
			A = self.sigmoid(z)
			#update weight dengan gradient descent
			grad = np.dot(features.T, (A-y))/y.shape[0]
			self.weight -= self.alpha*grad
			#loss
		print("done training. Time: ", str(time.time()-start_time))

		#testing dan akurasi disini

