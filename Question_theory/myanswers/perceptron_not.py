import numpy as np
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

        print(dt)
        print(dy)
        dt[dt * dy >= 0] = 0

        En = np.dot(dt, x)
        self.weight += En * self.lr

        is_changed = len(np.where(dt != 0)[0]) > 0

        return is_changed

xs = np.array([[0], [1]], dtype=np.float32)
ts = np.array([[1], [0]], dtype=np.float32)
lr = 0.1

_xs = np.hstack([xs, [[1] for _ in range(2)]])

w = np.random.normal(0., 1, (2))
 
nn = NN(_xs, lr)
ite = 0

while True:
    nn.forward(_xs)
    is_changed = nn.train(_xs, ts)
    ite += 1
    print("iteration: " + str(ite) + " y >> " + str(nn.y))
    if not(is_changed):
        break

print("training finished")
print("weight >> " + str(nn.weight))
for i in range(4):
    print("in >> " + str(_xs[i]) + " y >> " + str(nn.y[i]))



