'''
3.6.3 多層パーセプトロン
'''

import numpy as np


class MLP(object):
    '''
    多層パーセプトロン
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Layer(input_dim=input_dim,
                        output_dim=hidden_dim,
                        activation=sigmoid,
                        dactivation=dsigmoid)

        self.l2 = Layer(input_dim=hidden_dim,
                        output_dim=output_dim,
                        activation=sigmoid,
                        dactivation=dsigmoid)

        self.layers = [self.l1, self.l2]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y


class Layer(object):
    '''
    層間の結合
    '''
    def __init__(self, input_dim, output_dim,
                 activation, dactivation):
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
        self.b = np.random.normal(loc=0.0, scale=0.1, size=(output_dim,))

        self.activation = activation
        self.dactivation = dactivation

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self._input = x
        self._pre_activation = np.matmul(x, self.W) + self.b
        return self.activation(self._pre_activation)

    def backward(self, delta, W):
        delta = self.dactivation(self._pre_activation) \
                * np.matmul(delta, W.T)
        return delta

    def compute_gradients(self, delta):
        dW = np.matmul(self._input.T, delta)
        db = np.matmul(np.ones(self._input.shape[0]), delta)

        return dW, db


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


if __name__ == '__main__':
    np.random.seed(123)
    '''
    1. データの準備
    '''
    # XOR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    t = np.array([[0], [1], [1], [0]])

    '''
    2. モデルの構築
    '''
    model = MLP(2, 2, 1)

    '''
    3. モデルの学習
    '''
    def compute_loss(t, y, eps=1e-8):
        y = np.clip(y, eps, 1 - eps)
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()

    def train_step(x, t):
        y = model(x)
        grads = []
        W_prev = None
        for i, layer in enumerate(model.layers[::-1]):
            if i == 0:
                delta = y - t
            else:
                delta = layer.backward(delta, W_prev)

            dW, db = layer.compute_gradients(delta)
            grads.append((layer, dW, db))
            W_prev = layer.W
        for layer, dW, db in grads:
            layer.W = layer.W - 0.5 * dW
            layer.b = layer.b - 0.5 * db
        loss = compute_loss(t, y)
        return loss

    epochs = 1000

    for epoch in range(epochs):
        train_loss = train_step(x, t)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch,
                train_loss
            ))

    '''
    4. モデルの評価
    '''
    for input in x:
        print('{} => {:.3f}'.format(input, model(input)[0]))
    print("隠れ層の重み：")
    print(f"[w11,w21] = {model.l1.W[0]}, [b1] = {model.l1.b[0]}")
    print(f"[w12,w22] = {model.l1.W[1]}, [b2] = {model.l1.b[1]}")
    print("出力層の重み：")
    print(f"[v1,v2] = {model.l2.W[:,0]}, [c] = {model.l2.b[0]}")