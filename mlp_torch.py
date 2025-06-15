import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
t = torch.tensor([[0.], [1.], [1.], [0.]])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        # 出力には活性化を付けず、BCEWithLogitsLossと組み合わせる

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        return x

if __name__ == '__main__':
    # 再現性のためのシード設定
    torch.manual_seed(123)

    # モデル構築
    model = MLP(2, 2, 1)
    # Xavier 初期化
    nn.init.xavier_uniform_(model.l1.weight)
    nn.init.zeros_(model.l1.bias)
    nn.init.xavier_uniform_(model.l2.weight)
    nn.init.zeros_(model.l2.bias)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, nesterov=True)

    # 学習ループ
    epochs = 1000
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(x)
        loss = criterion(logits, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs:
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

    print("\n隠れ層の重み：")
    print(w1, b1)
    print("出力層の重み：")
    print(w2, b2)
