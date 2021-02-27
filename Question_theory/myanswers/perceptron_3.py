import numpy as np
#import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
np.random.seed(0)

class NN:

    def __init__(self, x, lr):
        #self.weight = np.random.normal(0., 1,  [x.shape[1], 1])
        self.weight = np.random.normal(0., 1,  x.shape[1])
        self.b = 1.
        self.lr = lr
        self.y = np.zeros(x.shape[0])

    def forward(self, x):
        self.y = np.dot(x, self.weight)

    def train(self, x, t):
        is_changed = False
        # t == y となってれば学習しなくていい
        # ただし，yは0以上で1, 0未満で-1とみなす
        dy = self.y.copy()
        dt = t.copy()
        # yとtの符号が同じなら学習しなくていいので，0にする

        dt[dt * dy >= 0] = 0

        En = np.dot(dt, x)
        self.weight += En * self.lr

        is_changed = len(np.where(dt != 0)[0]) > 0

        return is_changed

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([(-1), (-1), (-1), (1)], dtype=np.float32)
lr_1 = 0.1
lr_2 = 0.01
_xs = np.hstack([xs, [[1] for _ in range(4)]])


nn_1 = NN(_xs, lr_1)
nn_2 = NN(_xs, lr_2)
ite = 0

weight_0 = np.zeros(0)
weight_1 = np.zeros(0)
weight_2 = np.zeros(0)
ites = np.zeros(0)

while True:
    nn_1.forward(_xs)
    is_changed = nn_1.train(_xs, ts)
    weight_0 = np.append(weight_0, nn_1.weight[0])
    weight_1 = np.append(weight_1, nn_1.weight[1])
    weight_2 = np.append(weight_2, nn_1.weight[2])
    ite += 1
    ites = np.append(ites, ite)
    #print("iteration: " + str(ite) + " y >> " + str(nn_1.y))
    if not(is_changed):
        break

plt.plot(ites, weight_0)
plt.plot(ites, weight_1)
plt.plot(ites, weight_2)
'''
print("training finished")
print("weight >> " + str(nn.weight))
for i in range(4):
    print("in >> " + str(_xs[i]) + " y >> " + str(nn.y[i]))
'''

ite = 0
weight_0 = np.zeros(0)
weight_1 = np.zeros(0)
weight_2 = np.zeros(0)
ites = np.zeros(0)

while True:
    nn_2.forward(_xs)
    is_changed = nn_2.train(_xs, ts)
    weight_0 = np.append(weight_0, nn_2.weight[0])
    weight_1 = np.append(weight_1, nn_2.weight[1])
    weight_2 = np.append(weight_2, nn_2.weight[2])
    ite += 1
    ites = np.append(ites, ite)
    np.append(ites, ite)

    #print("iteration: " + str(ite) + " y >> " + str(nn_2.y))
    if not(is_changed):
        break

plt.plot(ites, weight_0)
plt.plot(ites, weight_1)
plt.plot(ites, weight_2)
#plt.show()
plt.savefig("perceptron_3.png")
