import numpy as np
import matplotlib.pyplot as plt

# 線形回帰器の定義
class LinearRegression():
    def __init__(self, n_feature):
        self.w = np.random.normal(n_feature)
        self.b = 0.

    def predict(self, x):
     return self.w * x + self.b
    
    def loss(self, x, y):
        N = x.shape[0]
        y_pred = self.predict(x)
        MSE = (1/N) * ((y_pred - y) ** 2).sum()
        return MSE

    def step(self, x, y, lr):
        N = x.shape[0]
        y_pred = self.predict(x)
        error = y_pred - y
        dw = (2/N) * (error * x).sum()
        db = (2/N) * error.sum()
        self.w -= lr * dw
        self.b -= lr * db

    def fit(self, x, y, lr=0.01, epoch=200):
        history = []
        for e in range(epoch):
            MSE_data = self.loss(x, y)
            history.append(MSE_data)
            self.step(x, y, lr)
            if e % 20 == 0:
                print(f"Epoch {e:3d}: loss={MSE_data:.4f}")
        print(f"Trained Parameters: w={self.w:.4f}, b={self.b:.4f}")

        return history, self.w, self.b

# データの準備
x_data = np.linspace(-10, 10, 200)
epsilon = np.random.normal(0, 2, 200)
a = 3.5
b = 1.2

y = a * x_data + b + epsilon

# 学習
model = LinearRegression(1)
MSE, w_pred, b_pred = model.fit(x_data, y, epoch=200, lr=0.001)

# 結果の出力
plt.figure()
plt.plot(MSE)
plt.savefig("MSE_graph.png")
y_pred = w_pred * x_data + b_pred
plt.figure()
plt.scatter(x_data, y, 10, c="Green")
plt.plot(x_data, y_pred, color="red")
y_real = a * x_data + b
plt.plot(x_data, y_real, color="Blue")
plt.savefig("output.png")