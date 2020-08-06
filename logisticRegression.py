import numpy as np
class logReg:
    '''
        Logistic regression classifier.
        Dito Aldi Soekarno Putra 1151500054
        Informatika Institut Teknologi Indonesia
        
        Referensi:
        [1] B. Tan, “How to Classify Cat Pics with a Logistic Regression Model,” Medium, 03-Apr-2020. [Online].
            Available: https://towardsdatascience.com/classifying-cat-pics-with-a-logistic-regression-model-e35dfb9159bb. [Accessed: 06-Aug-2020].

        [2] A. Pant, “Introduction to Logistic Regression,” Medium, 22-Jan-2019. [Online].
            Available: https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148. [Accessed: 06-Aug-2020].
        
        [3] S. Swaminathan, “Logistic Regression - Detailed Overview,” Medium, 15-Mar-2019. [Online].
            Available: https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc. [Accessed: 06-Aug-2020].
    '''
    def __init__(self, epoch = 2000, alpha = 0.01):
        self.epoch = epoch
        self.alpha = alpha
    
    #inisialisasi weight dan bias
    def initialize_with_zeros(self, dim):
        w = np.zeros((1, dim))
        b = 0
        return w, b

    #aktivasi sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #fungsi cost
    def cost(self, m, Y, A):
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
        cost = np.squeeze(cost)
        print ("Cost: ", cost)
        return cost

    #fungsi propagasi forward dan backward
    def propagate(self, w, b, X, Y):
        m = X.shape[0]
        #forward propagation
        A = self.sigmoid(np.dot(w, X.T) + b)
        #backward propagation
        dw = (1/m) * np.dot(X.T, (A-Y).T)
        db = (1/m) * np.sum(A-Y)
        grads = {"dw": dw,
                "db": db}

        return grads, A
    
    #fungsi optimasi
    def optimize(self, iter, w, b, X, Y):
        if iter == "final":
            message = "final"
        else:
            message = iter+1
        for i in range(self.epoch):
            print("\nFine tuning layer number:", message)
            print("iteration:",i)
            grads, A = self.propagate(w, b, X, Y) #hitung aktivasi dan gradien
            dw = grads["dw"]
            db = grads["db"]
            #optimasi weight dan bias dengan gradient decent
            w = w - self.alpha * dw.T #weight baru
            b = b - self.alpha * db #bias baru
            self.cost(Y.shape[1],Y,A) #hitung cost
        params = w
        bias = b
        grads = {"dw": dw,
                "db": db}

        return params, bias, A, grads

    #fungsi prediksi
    def predict(self, w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(1, X.shape[1])
        A = self.sigmoid(np.dot(w, X.T) + b) #hitung probabilitas aktivasi data inputan
        for i in range(A.shape[1]):
                Y_prediction = A[0,i] #probabilitas untuk prediksi kelas

        return Y_prediction

    #fungsi latih
    def fit(self, X_train, Y_train, X_test, Y_test):
        #inisialisasi parameter dan bias
        w, b = self.initialize_with_zeros(X_train.shape[1])
        # optimasi dengan gradient descent
        parameters, bias, A, grads = self.optimize("final", w, b, X_train, Y_train)
        w = parameters
        b = bias
        #prediksi training dan testing
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)
        #akurasi
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        #detail hasil training
        d = {
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "alpha" : self.alpha,
            "epoch": self.epoch}

        return w, b
