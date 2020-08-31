#impor library yang dibutuhkan
from rbm import RBM
from logisticRegressionSGD import logReg
import numpy as np
import time
class DBN:
    '''
        Deep Belief Network
        Dito Aldi Soekarno Putra 1151500054
        Informatika Institut Teknologi Indonesia

        Referensi:
        [1] “Deep Belief Networks¶,” Deep Belief Networks - DeepLearning 0.1 documentation. [Online].
            Available: http://deeplearning.net/tutorial/DBN.html. [Accessed: 06-Aug-2020].

        [2] C. Nicholson, “Deep-Belief Networks,” Pathmind. [Online].
            Available: https://pathmind.com/wiki/deep-belief-network. [Accessed: 06-Aug-2020].

        [3] I. C. Labs, “Deep Belief Networks - all you need to know,” Medium, 08-Aug-2018. [Online].
            Available: https://medium.com/@icecreamlabs/deep-belief-networks-all-you-need-to-know-68aa9a71cc53. [Accessed: 06-Aug-2020].
        
        [4] C. Li, Y. Wang, X. Zhang, H. Gao, Y. Yang, and J. Wang,
            “Deep Belief Network for Spectral–Spatial Classification of Hyperspectral Remote Sensor Data,” Sensors, vol. 19, no. 1, p. 204, 2019.
    '''
    def __init__(self, n_nodes, rbm_epoch, max_epoch, alpha, threshold):
        self.n_nodes = n_nodes
        self.rbm_epoch = rbm_epoch
        self.rbm_layers = []
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.threshold = threshold
        self.params = []
        self.hidden_layer = []
        self.visible_layer = []
        self.lr_layer = None
        self.rbm_inference = []
        self.bias_node = None
        self.start_pretrain_time = 0
        self.stop_pretrain_time = 0
        self.start_finetune_time = 0
        self.total_ft_epoch = max_epoch

    #fungsi transform untuk mencari hidden layer data inputan
    def transform(self, X):
        for i in range(3):
            X = self.rbm_layers[0].activation(self.params[i], self.rbm_inference[i], X)
        return X

    def build_model(self):
        n_h=[]
        n_v = [self.n_nodes]
        n_hPercentage = [60, 30, 10]
        for i in n_hPercentage:
            nhid = int(np.round(self.n_nodes*(i/100)))
            n_h.append(nhid)
            n_v.append(nhid)
        n_v.pop()
        self.bias_node = 1
        #membangun model
        #layer RBM
        for n in range(3):
            rbm_layer = RBM(epoch=self.rbm_epoch, n_visible = n_v[n], n_hidden = n_h[n], alpha=0.01)
            self.rbm_layers.append(rbm_layer)
        self.fine_tuner = logReg(1, self.alpha, self.threshold)
        #layer logistic regression
        self.lr_layer = logReg(self.max_epoch, self.alpha, self.threshold)
        #print(n_v, n_h)

    #fungsi pre-training dengan 3 RBM
    def pre_train(self, X):
        self.start_pretrain_time = time.time()
        self.visible_layer.append(X)
        for rbm in self.rbm_layers:
            print("X", X.shape[0])
            X, w, rbm_bias = rbm.fit(X)
            self.params.append(w)
            self.rbm_inference.append(rbm_bias)
            self.hidden_layer.append(X)
            self.visible_layer.append(X)
        self.visible_layer.pop()
        w_class = np.asarray(np.random.normal(0, 0.01,(self.hidden_layer[2].shape[1],1)))
        self.params.append(w_class)
        self.stop_pretrain_time = time.time()
        #print("rbm bias:", self.rbm_inference)
        return X

    #fungsi fine-tuning dengan supervised gradient decent dan klasifikasi dengan logistic regression
    def fine_tune(self, y, X_test, y_test):
        self.start_finetune_time = time.time()
        print("start fine-tuning step\n")
        for e in range(self.max_epoch):
            print("\nFin-tuning iteration number: ", e)
            avg_cost = 0
            for i in range (4):
                #optimasi parameter dan bias
                if i < 3:
                    infereces_reshaped = self.rbm_inference[i].reshape(1,self.rbm_inference[i].shape[0])
                    bias = infereces_reshaped
                    visible = self.visible_layer[i]
                else:
                    bias = self.bias_node
                    visible = self.hidden_layer[2]
                params, inferences, hiddens, grads, cost = self.fine_tuner.optimize(i, self.params[i], bias, visible, y)
                #print("new w shape: ", self.params[i].shape)
                self.params[i] = params
                if i < 3:
                    self.rbm_inference[i] = inferences.reshape(inferences.shape[1])
                    self.hidden_layer[i] = hiddens
                else:
                    self.bias_node = inferences
                #memasukkan hidden dan visible layer baru
                if i < 2:
                    self.visible_layer[i+1] = hiddens
            avg_cost = cost
            print("cost: ", avg_cost)
            if avg_cost <= self.threshold:
                self.total_ft_epoch = e
                break
        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(self.visible_layer[0])
        train_acc = np.sqrt(np.mean(np.square(Y_prediction_test-y)))
        test_acc = np.sqrt(np.mean(np.square(Y_prediction_train-y_test)))
        print("\ntrain accuracy: ", train_acc)
        print("test accuracy: ", test_acc)
        #pelatihan klasifikasi dengan logistic regression
        '''
        
        print("\nstart training classification\n")
        lr_w, self.bias_node = self.lr_layer.fit(self.hidden_layer[2], y, x_test, y_test)
        self.params.append(lr_w)
        '''

    #fungsi latih
    def fit(self, X, y, x_test, y_test):
        start_total_time = time.time()
        #penyusun model
        self.build_model()
        #tahap pre-training
        self.pre_train(X)
        #tahap fine-tuning
        self.fine_tune(y, x_test, y_test)
        print("\nNumber of RBM epoch: ", self.rbm_epoch)
        print("Number of fine-tuning epoch: ", self.total_ft_epoch)
        print("\nTotal training time:" + str((time.time() - start_total_time)/3600) + " hrs")
        print(" - pre-training:" + str((self.stop_pretrain_time - self.start_pretrain_time)/3600) + " hrs")
        print(" - fine-tuning:" + str((time.time() - self.start_finetune_time)/3600) + " hrs")
    
    #fungsi prediksi kelas
    def predict(self, X):
        #pencari hidden layer dari data input
        transformed = self.transform(X)
        #transformed = transformed.reshape(transformed.shape[0], 1)
        #probabilitas kelas
        prediction = self.lr_layer.predict(self.params[3], self.bias_node, transformed)
        
        return prediction