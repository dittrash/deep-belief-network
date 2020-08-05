#impor library yang dibutuhkan
from rbm import RBM
from logreg import LogisticReg
import numpy as np
class DBN:
    def __init__(self, n_nodes, rbm_epoch, max_epoch, alpha):
        self.n_nodes = n_nodes
        self.rbm_epoch = rbm_epoch
        self.rbm_layers = []
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.params = []
        self.hidden_layer = []
        self.visible_layer = []
        self.lr_layer = None

    def build_model(self):
        n_h=[]
        n_v = [self.n_nodes]
        n_hPercentage = [60, 20, 10]
        for i in n_hPercentage:
            nhid = int(np.round(self.n_nodes*(i/100)))
            n_h.append(nhid)
            n_v.append(nhid)
        n_v.pop()
        
        #membangun model
        for n in range(3):
            rbm_layer = RBM(epoch=self.rbm_epoch, n_visible = n_v[n], n_hidden = n_h[n], alpha=self.alpha)
            self.rbm_layers.append(rbm_layer)
        self.lr_layer = LogisticReg(n_h[2], self.max_epoch, self.alpha)
        print(n_v, n_h)
		
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))


    def pre_train(self, X):
        self.visible_layer.append(X)
        for rbm in self.rbm_layers:
            print("X", X.shape[0])
            X, w = rbm.fit(X)
            self.params.append(w)
            self.hidden_layer.append(X)
            self.visible_layer.append(X)
        self.visible_layer.pop()
        print(self.params)
        return X

    def sgd(self, visible, hidden, y):
        for i in range(3):
            hidden[i] = self.rbm_layers[0].sigmoid(np.dot(visible[i],self.params[i]))
            #print("\nhidden",i, self.hidden_layer[i])
            gradient = np.dot(visible[i].T, (hidden[i]-y.T))/y.shape[1]
            self.params[i] -= self.alpha*gradient
        return self

    def fine_tune(self, h, y):
        self.sgd(self.visible_layer, self.hidden_layer, y)
        cost = np.sum(y.T * np.log(h) + (1-y.T) * np.log(1-h))/-y.shape[1]
        print("cost:", cost)

    def fit(self, X, y):
        #tahap pretraining
        self.build_model()
        self.pre_train(X)
        for epoch in range(self.max_epoch):
            print("\nFine tuning iteration:",epoch)
            self.fine_tune(self.hidden_layer[2], y)
    
    def predict(self, X):
        for i in range(3):
            X = np.dot(X, self.params[i])
            X = self.sigmoid(X)
        return X