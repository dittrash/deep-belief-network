#impor library yang diperlukan
import numpy as np

class RBM:
#inisialisasi variabel
    def __init__(self,
        epoch,
        n_visible,
        n_hidden,
        alpha):
            self.n_visible = n_visible
            self.n_hidden= n_hidden
            self.alpha = alpha
            self.epoch = epoch
            self.np_rng = np.random.RandomState(1234)
            #bobot
            self.weight = np.asarray(np.random.normal(0, 0.01,(self.n_visible, self.n_hidden)))
            #bias
            self.hb = np.ones(self.n_hidden)
            self.vb = np.ones(self.n_visible)
	
	#fungsi sigmoid
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    #latih
    def activation(self, X):
        #menghitung aktivasi P(h=1|v)
        #dan P(v=1|h)
        #return hPpos, hPneg 
        z = np.dot(X, self.weight)
        z += self.hb
        return self.sigmoid(z)

    def sample_hidden(self, hP, dp):
        #menghitung distribusi P(h|v)
        #return hPos
        rng = np.random.rand(dp, hP.shape[1])
        #hP = self.activation(hP)
        return (rng < hP)

    def sample_visible(self, hPos, dp):
        #menghitung distribusi P(v|h)
        #return vNeg
        vP = np.dot(hPos, self.weight.T)
        vP += self.vb
        vP = self.sigmoid(vP)
        rng = np.random.rand(dp, vP.shape[1])
        return (rng < vP), vP

    def pos_grad(self, X, hPos):
        #menghitung graden positif
        #return pos_grad
        return np.dot(X.T, hPos)
        
    def neg_grad(self, vNeg, hPneg):
        #menghitung gradien negative
        #return neg_grad
        return np.dot(vNeg.T, hPneg)

    def fit(self, X):
        #fungsi latih dan update weight & bias
        dp = X.shape[0] #jumlah datapoint
        for i in range(self.epoch):
            hPpos = self.activation(X)
            hPos = self.sample_hidden(hPpos, dp)
            vNeg, vPneg = self.sample_visible(hPos, dp)
            hPneg = self.activation(vPneg)
            pos_grad = self.pos_grad(X, hPos)
            neg_grad = self.neg_grad(vNeg, hPneg)
            #update weight dgn CD
            cd = (pos_grad-neg_grad)/dp
            self.weight += self.alpha*cd

            #update bias vb & hb
            self.hb += self.alpha*(hPpos.sum(axis=0)-hPneg.sum(axis=0))
            self.vb += self.alpha*(X.sum(axis=0)-vNeg.sum(axis=0))
            #error
            error = np.mean(np.square(vPneg-X))
            print("\nRBM iteration", i)
            print("error: ", error)
        final_w = self.weight
        return hPpos, final_w
