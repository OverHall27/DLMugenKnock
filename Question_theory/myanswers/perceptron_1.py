import numpy as np

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)

np.random.seed(0)
# weight は初期値をランダム値にしてる
w = np.random.normal(0., 1, (3))

# add bias -> 重みを1倍するため．x1×w1 + x2×w2 + 1×w3 という形にするための 1
_xs = np.hstack([xs, [[1] for _ in range(4)]])
print(_xs)
   
print("weight >> " + str(w))
for i in range(4):
    ys = np.dot(w, _xs[i])
    print("in >> " + str(_xs[i]) + " y >> " + str(ys))



