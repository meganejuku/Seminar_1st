import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# モデルの定義
class SimplePerseptron():
    def __init__(self, input_dim, lr = 0.01):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim, ))
        self.b = 0.
        self.lr = lr

    def forward(self, x):
        y = step(np.matmul(self.w, x) + self.b)
        return y
    
    def compute_deltas(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = delta * self.lr * x
        db = delta * self.lr
        return dw, db
def step(x):
    return 1 * (x > 0)

if __name__ == "__main__":
    # データの用意
    d = 2 
    N = 50

    mean = 4.5

    x1 = np.random.randn(N//2, d) + np.array([0, 0])
    x2 = np.random.randn(N//2, d) + np.array([mean, mean])
    x = np.vstack((x1, x2))
    t = np.hstack((np.zeros(N//2), np.ones(N//2)))

    # コンストラクタを取得
    model = SimplePerseptron(d)

    history = []
    def train_epoch():
        converged = True
        for xi, ti, in zip(x, t):
            dw, db = model.compute_deltas(xi, ti)
            if (dw == 0).all() and db == 0:
                ...
            else:
                converged = False
            model.w -= dw
            model.b -= db
            history.append((model.w.copy(), model.b))
        return converged
    
    for epoch in range(100):
        if train_epoch():
            print(f"converged at epoch {epoch}")
            break
    frame_late = 60
    hold_frames = frame_late  # FPS=10 → 10フレームで約1秒
    history.extend([history[-1]] * hold_frames)

    fig, ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1],s=20, c=t, cmap='bwr', alpha=0.6)
    ax.set_xlim(-3, mean+3)
    ax.set_ylim(-3, mean+3)    
    line, = ax.plot([], [], 'k-', lw=2)
    ax.set_title("Perceptron Decision Boundary")
    def init():
        line.set_data([], [])
        return (line,)
    def update(frame):
        w, b = history[frame]
        # 境界線: w0*x + w1*y + b = 0  →  y = -(w0*x + b)/w1
        xs = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        ys = -(w[0] * xs + b) / w[1]
        line.set_data(xs, ys)
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        init_func=init, blit=True, interval=100
    )
    ani.save("train_GIF.gif", writer='pillow', fps=frame_late)
    plt.show()
    print("Saved GIF → train_GIF.gif")


    x1_data = []
    x2_data = []

    plot_x1 = 10.
    plot_x2 = (-model.b - (model.w[0] * plot_x1)) / model.w[1]
    x1_data.append(plot_x1)
    x2_data.append(plot_x2)

    plot_x1 = -10.
    plot_x2 = (-model.b - (model.w[0] * plot_x1)) / model.w[1]
    x1_data.append(plot_x1)
    x2_data.append(plot_x2)

    plt.plot(x1_data, x2_data)
    plt.savefig("result.png")
    plt.show()
