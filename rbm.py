#impor library yang diperlukan
import numpy as np

class RBM:
    '''
        Restricted Boltzmann Machine.
        Dito Aldi Soekarno Putra 1151500054
        Informatika Institut Teknologi Indonesia

        Referensi:
        [1] S. Deb, “Restricted Boltzmann Machine Tutorial: Deep Learning Concepts,” Edureka!, 21-May-2020. [Online].
            Available: https://www.edureka.co/blog/restricted-boltzmann-machine-tutorial/. [Accessed: 06-Aug-2020].

        [2] “Restricted Boltzmann Machines (RBM)¶,” DeepLearning 0.1 documentation. [Online].
            Available: http://deeplearning.net/tutorial/rbm.html. [Accessed: 06-Aug-2020].

        [3] A. Sharma, “Restricted Boltzmann Machines - Simplified,” Medium, 06-Dec-2018. [Online].
            Available: https://towardsdatascience.com/restricted-boltzmann-machines-simplified-eab1e5878976. [Accessed: 06-Aug-2020].
        
        [4] M. Nayak, “An Intuitive Introduction Of Restricted Boltzmann Machine (RBM),” Medium, 18-Apr-2019. [Online].
            Available: https://medium.com/datadriveninvestor/an-intuitive-introduction-of-restricted-boltzmann-machine-rbm-14f4382a0dbb. [Accessed: 06-Aug-2020].
    '''
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

    #fungsi aktivasi
    def activation(self, w, b, X):
        #menghitung aktivasi P(h=1|v)
        #dan P(v=1|h)
        #return hPpos, hPneg 
        z = np.dot(X, w)
        z += b
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
            hPpos = self.activation(self.weight, self.hb, X)
            hPos = self.sample_hidden(hPpos, dp)
            vNeg, vPneg = self.sample_visible(hPos, dp)
            hPneg = self.activation(self.weight, self.hb, vPneg)
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
        final_hb = self.hb
        return hPpos, final_w, final_hb
