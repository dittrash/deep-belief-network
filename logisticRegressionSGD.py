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
    def __init__(self, epoch = 2000, alpha = 0.01, threshold=0.011):
        self.epoch = epoch
        self.lr_epoch = epoch
        self.alpha = alpha
        self.threshold = threshold
        self.total_lr_epoch = epoch
    
    #inisialisasi weight dan bias
    def initialize_with_zeros(self, dim):
        w = np.zeros((dim,1))
        b = 0
        return w, b

    #aktivasi sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #fungsi cost
    def cost(self, Y, A):
        cost = -np.sum(Y * np.log(A) + (1 - Y)* np.log(1-A))
        return cost

    #fungsi propagasi forward dan backward
    def propagate(self, w, b, X, Y):
        #forward propagation
        A = self.sigmoid(np.dot(X, w) + b)
        #backward propagation
        dw = np.dot(X.T, (A-Y))
        db = (A-Y)
        grads = {"dw": dw,
                "db": db}

        return grads, A
    
    #fungsi optimasi
    def optimize(self, iter, w, b, X, Y):
        stopped_at = 0
        if iter == "final":
            message = "final"
        for i in range(self.epoch):
            #randomize input
            h = np.zeros(shape=(X.shape[0], w.shape[1]))
            datacost = 0
            indices = np.arange(Y.T.shape[0])
            np.random.shuffle(indices)
            xRand = X[indices]
            yRand = Y.T[indices]
            for j in range(len(X)):
                data_index = indices[j]
                print("data processed: ", f'{j}\r', end="")
                x_reshaped = xRand[j].reshape(1,len(xRand[j]))
                #print("xRand shape", x_reshaped.shape)
                y_reshaped = yRand[j].reshape(1,1)
                grads, A = self.propagate(w, b, x_reshaped, y_reshaped) #hitung aktivasi dan gradien
                dw = grads["dw"]
                db = grads["db"]
                
                #optimasi weight dan bias dengan gradient decent
                w = w - self.alpha * dw #weight baru
                b = b - self.alpha * db #bias baru
                datacost += self.cost(yRand[j],A) #hitung cost
                h[data_index] = A.reshape(A.shape[1])
                #print(datacost)
            avg_cost = datacost/Y.shape[1]
            if iter == "final":
                print("LR epoch:", message)
                print("iteration:",i+1)
                print ("Cost: ", avg_cost)
                print("\n")
                if avg_cost <= self.threshold:
                    self.total_lr_epoch = i
                    break
        params = w
        bias = b
        grads = {"dw": dw,
                "db": db}
        return params, bias, h, grads, avg_cost

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
        parameters, bias, h, grads, cost = self.optimize("final", w, b, X_train, Y_train)
        w = parameters
        b = bias
        #prediksi training dan testing
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)
        #akurasi
        print("\nNumber of LR epoch:", self.total_lr_epoch)
        print(" - train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print(" - test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        #detail hasil training
        d = {
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "alpha" : self.alpha,
            "epoch": self.epoch}

        return w, b
