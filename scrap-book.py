import numpy as np
k = []
m = np.array([[1,1,1,1],[0,0,0,0],[9,9,9,9]])
k.append(m)
z = np.array([[1,1,1],[0,0,0],[5,5,5]])
y = np.array([[1],[2],[3]])
k.append(z)
for x in range(len(k)):
    print("iteration", x)
    for q in range(2):
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        a = y[indices]
        b = k[x][indices]
        print("kx\n", k[x])
        print("b\n", b)
        for m in range(len(a)):
            print(indices[m])
            print("b[q]", b[m])

