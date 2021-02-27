import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class NN:

    def __init__(self, x, lr):
        self.weight = np.random.normal(0., 1,  x.shape[1])
        self.z1 = x
        self.b = 1.
        self.lr = lr
        self.y = np.zeros(x.shape[0])

    def forward(self, x):
        self.y = sigmoid(np.dot(x, self.weight))

    def train(self, x, t):

        En = -(t - self.y) * self.y * (1. - self.y)
        grad_w = np.dot(self.z1.T, En)
        self.weight -= self.lr * grad_w
 
def sigmoid(x):
    return 1. / (1. + np.exp(-x)) 

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([0, 0, 0, 1], dtype=np.float32)
lr = 0.1
_xs = np.hstack([xs, [[1] for _ in range(4)]])


nn = NN(_xs, lr)
print(nn.weight)
for _ in range(5000):
    nn.forward(_xs)
    nn.train(_xs, ts)

for i in range(4):
    print("in >> " + str(_xs[i]) + " y >> " + str(nn.y[i]))
print(nn.weight)
