import numpy as np
import matplotlib.pyplot as plt

x_init = np.array([[12],
                   [70],
                   [24],
                   [60],
                   [40]])

x_new = np.ones((len(x_init), 1))
x = np.concatenate((x_init, x_new), axis=1)

y = np.array([[1],
              [0],
              [1],
              [0],
              [1]])

w = np.array([[1],
              [1]])

lr = 0.01

for i in range(len(x)):
    n = np.matmul(x[i], w)
    y_pred = 1 if n >= 0 else 0
    dw = lr * (y[i] - y_pred) * x[i].reshape(-1,1)
    w = w + dw

print("weights:", w)

# test distance
distance = 51

test = np.array([distance, 1])

n = np.matmul(test, w)
prediction = 1 if n >= 0 else 0

if prediction == 1:
    print("Keyboard distance", distance, "is GOOD")
else:
    print("Keyboard distance", distance, "is BAD")

# plot 
plt.scatter(x_init, y)

boundary = -w[1] / w[0]
plt.axvline(boundary, linestyle='--')

plt.xlabel("distance")
plt.ylabel("good(1) / bad(0)")
plt.show()