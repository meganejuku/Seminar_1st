import numpy as np

class SimplePerceptron():

    def __init__(self, input_dim, lr=1.0):
        self.input_dim = input_dim
        self.lr = lr
        self.w = np.random.normal(size=(input_dim,))
        self.b = 0.

    def step(self, x):
        return 1 * (x > 0)
    
    def forward(self, x):
        y = self.step(np.dot(self.w , x) + self.b)
        return y
    
    def adjust_loss(self, x, correct):
        y = self.forward(x)
        delta = correct - y
        dw = self.lr * delta * x
        db = self.lr * delta

        self.w += dw
        self.b += db

        return (delta != 0)
    
if __name__ == "__main__":
    x_in = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1]),
    ]


    correct_gate = [0, 0, 0, 1]
    

    model = SimplePerceptron(input_dim=2, lr=0.1)

    for epoch in range(50):
        any_update = False
        for x, t in zip(x_in, correct_gate):
            judge = model.adjust_loss(x, t)
            any_update |= judge
        print(f"epoch : {epoch + 1}, w = {model.w},b = {model.b}")
        
        if not any_update:
            break
    
    for x, t in zip(x_in, correct_gate):
        pred = model.forward(x)
        print(f"入力{x} -> 予想{pred} / 正解{t}")