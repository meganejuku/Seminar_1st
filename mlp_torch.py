import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(123)

    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    t = torch.tensor([[0.], [1.], [1.], [0.]])
    # モデル構築
    model = MLP(2, 2, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, nesterov=True)


    epochs = 1000
    # 学習ループ
    model.train()

    for epoch in range(epochs):
        logits = model(x)
        loss = criterion(logits, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch: {epoch}, loss: {loss.item():.3f}")

    # 評価
    model.eval()

    with torch.no_grad():
        logits = model(x)
        preds = torch.sigmoid(logits)

    print("\n結果：")
    for inp, out in zip(x, preds):
        print(f"{inp.tolist()} => {out.item():.3f}")

    # 重みの表示
    w1, b1 = model.l1.weight.data, model.l1.bias.data
    w2, b2 = model.l2.weight.data, model.l2.bias.data

    print("隠れ層の重み：")
    print(f"[w11,w21] = {w1[0]}, [b1] = {b1[0]}")
    print(f"[w12,w22] = {w1[1]}, [b2] = {b1[1]}")
    print("出力層の重み：")
    print(f"[v1,v2] = {w2[0]}, [c] = {b2[0]}")