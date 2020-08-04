from dbn import DBN
import numpy as np
model = DBN(n_nodes=6,rbm_epoch=10,max_epoch=5, alpha=0.1)
X = np.array([[0.2157, 0.1255, 0.4039, 1.0, 0.0941, 0.2550],
                [0.1686, 0.9529, 0.0824, 0.0980, 1.0, 0.3529],
                [0.3529, 0.0824, 0.4275, 1.0, 0.1255, 0.2941],
                [0.1255, 1.0, 0.1216, 0.0471, 1.0, 0.2431]])
y = np.array([0,1,0,1])
model.fit(X, y)