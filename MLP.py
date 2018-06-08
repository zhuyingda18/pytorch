import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

mnist = fetch_mldata('MNIST original', data_home='./data/')
X, Y = mnist.data, mnist.target
X = X / 255.
Y = Y.astype("int")


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
train_y = np.eye(10)[train_y].astype(np.int32)
test_y = np.eye(10)[test_y].astype(np.int32)
train_n = train_x.shape[0]
test_n = test_x.shape[0]


class Sigmoid:
    def __init__(self):
        self.y = None

    def __call__(self, x):
        y = 1 / (1 + np.exp(-x))  # 順伝播計算
        self.y = y
        return y

    def backward(self):
        return self.y * (1 - self.y)  # 逆伝播計算


class ReLU:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return x * (x > 0)  # 順伝播計算

    def backward(self):
        return 1 * (self.x > 0)  # 逆伝播計算

class Softmax:
    def __init__(self):
        self.y = None

    def __call__(self, x):
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))  # ここで exp(x - x_max) を計算しよう
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)  # exp_x を用いて softmax を計算しよう
        self.y = y
        return y

def dropout1(x, level):
    if level < 0. or level >= 1  :
        raise Exception('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level
    sample =np.random.binomial(n=1 ,p=retain_prob ,size=x.shape  )
    x *= sample
    x /= retain_prob
    return x

class Linear:
    def __init__(self, in_dim, out_dim, activation,num):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation()
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None
        self.i = num

    def __call__(self, x):
        # 順伝播計算
        self.x = x * dropout[self.i][0]/dropout[self.i][1]  # 順伝播 Dropout
        u = np.dot(self.x, self.W) + self.b  # self.W, self.b, x を用いて u を計算しよう
        self.z = self.activation(u)
        return self.z

    def backward(self, dout):
        # 誤差計算
        dout = dout * dropout[self.i+1][0]/dropout[self.i+1][1] # 逆伝播 Dropout
        self.delta = dout * self.activation.backward()  # dout と活性化関数の逆伝播 (self.activation.backward()) を用いて delta を計算しよう
        dout = np.dot(self.delta, self.W.T)  # self.delta, self.W を用いて 出力 o を計算しよう

        # 勾配計算
        self.dW = np.dot(self.x.T, self.delta)  # dW を計算しよう
        self.db = np.dot(np.ones(len(self.x)), self.delta)  # db を計算しよう

        return dout


class MLP():
    def __init__(self, layers):
        self.layers = layers

    def train(self, x, t, lr):
        # 1. 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)  # 順伝播計算を順番に行い， 出力 y を計算しよう

        # 2. 損失関数の計算
        self.loss = np.sum(-t * np.log(self.y + 1e-7)) / len(x)

        # 3. 誤差逆伝播
        # 3.1. 最終層
        # 3.1.1. 最終層の誤差・勾配計算
        delta = (self.y - t) / len(self.layers[-1].x)
        self.layers[-1].delta = delta
        self.layers[-1].dW = np.dot(self.layers[-1].x.T, self.layers[-1].delta)
        self.layers[-1].db = np.dot(np.ones(len(self.layers[-1].x)), self.layers[-1].delta)
        dout = np.dot(self.layers[-1].delta, self.layers[-1].W.T)

        # 3.1.2. 最終層のパラメータ更新
        self.layers[-1].W -= lr * self.layers[-1].dW  # self.layers[-1].dW を用いて最終層の重みを更新しよう
        self.layers[-1].b -= lr * self.layers[-1].db  # self.layers[-1].db を用いて最終層のバイアスを更新しよう

        # 3.2. 中間層
        for layer in self.layers[-2::-1]:
            # 3.2.1. 中間層の誤差・勾配計算
            dout = layer.backward(dout)  # 逆伝播計算を順番に実行しよう

            # 3.2.2. パラメータの更新
            layer.W -= lr * layer.dW  # 各層の重みを更新
            layer.b -= lr * layer.db  # 各層のバイアスを更新

        return self.loss

    def test(self, x, t):
        # 性能をテストデータで調べるために用いる
        # よって，誤差逆伝播は不要
        # 順伝播 (train関数と同様)
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t * np.log(self.y + 1e-7)) / len(x)
        return self.loss

dp = [0,0.3,0.3,0]
dropout = []
dropout.append([np.random.binomial(n=1, p=1-dp[0], size=784),1-dp[0]])
dropout.append([np.random.binomial(n=1, p=1-dp[1], size=1000),1-dp[1]])
dropout.append([np.random.binomial(n=1, p=1-dp[2], size=500),1-dp[2]])
dropout.append([np.random.binomial(n=1, p=1-dp[3], size=200),1-dp[3]])
# Dropout matrix,nueral elimination

model = MLP([Linear(784, 1000, ReLU,0),
             Linear(1000, 500, ReLU,1),
             Linear(500, 200, ReLU,2),
             Linear(200, 10, Softmax,3)])


n_epoch = 50
batchsize = 100
lr = 0.7

for epoch in range(n_epoch):
    print('epoch %d | ' % epoch, end="")

    # 訓練
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(train_n)

    for i in range(0, train_n, batchsize):
        x = train_x[perm[i: i + batchsize]]
        t = train_y[perm[i: i + batchsize]]
        sum_loss += model.train(x, t, lr) * len(x)
        # model.y には， (N, 10)の形で，画像が0~9の各数字のどれに分類されるかの事後確率が入っている
        # そこで，最も大きい値をもつインデックスを取得することで，識別結果を得ることができる
        pred_y.extend(np.argmax(model.y, axis=1))

    loss = sum_loss / train_n

    # accuracy : 予測結果を1-hot表現に変換し，正解との要素積の和を取ることで，正解数を計算できる．
    accuracy = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_n
    print('Train loss %.3f, accuracy %.4f | ' % (loss, accuracy), end="")

    # テスト
    sum_loss = 0
    pred_y = []

    for i in range(0, test_n, batchsize):
        x = test_x[i: i + batchsize]
        t = test_y[i: i + batchsize]

        sum_loss += model.test(x, t) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))

    loss = sum_loss / test_n
    accuracy = np.sum(np.eye(10)[pred_y] * test_y) / test_n
    print('Test loss %.3f, accuracy %.4f' % (loss, accuracy))

# epoch 0 | Train loss 0.442, accuracy 0.8593 | Test loss 0.143, accuracy 0.9583
# epoch 1 | Train loss 0.112, accuracy 0.9663 | Test loss 0.258, accuracy 0.9263
# epoch 2 | Train loss 0.074, accuracy 0.9770 | Test loss 0.106, accuracy 0.9696
# epoch 3 | Train loss 0.056, accuracy 0.9819 | Test loss 0.082, accuracy 0.9769
# epoch 4 | Train loss 0.042, accuracy 0.9868 | Test loss 0.092, accuracy 0.9754
# epoch 5 | Train loss 0.033, accuracy 0.9893 | Test loss 0.092, accuracy 0.9770
# epoch 6 | Train loss 0.025, accuracy 0.9922 | Test loss 0.116, accuracy 0.9721
# epoch 7 | Train loss 0.022, accuracy 0.9931 | Test loss 0.088, accuracy 0.9777
# epoch 8 | Train loss 0.019, accuracy 0.9941 | Test loss 0.100, accuracy 0.9774
# epoch 9 | Train loss 0.016, accuracy 0.9947 | Test loss 0.092, accuracy 0.9797
# epoch 10 | Train loss 0.012, accuracy 0.9963 | Test loss 0.106, accuracy 0.9785
# epoch 11 | Train loss 0.013, accuracy 0.9957 | Test loss 0.089, accuracy 0.9794
# epoch 12 | Train loss 0.012, accuracy 0.9965 | Test loss 0.086, accuracy 0.9818
# epoch 13 | Train loss 0.008, accuracy 0.9972 | Test loss 0.098, accuracy 0.9805
# epoch 14 | Train loss 0.010, accuracy 0.9971 | Test loss 0.097, accuracy 0.9809
# epoch 15 | Train loss 0.012, accuracy 0.9963 | Test loss 0.089, accuracy 0.9820
# epoch 16 | Train loss 0.010, accuracy 0.9969 | Test loss 0.093, accuracy 0.9818
# epoch 17 | Train loss 0.004, accuracy 0.9988 | Test loss 0.094, accuracy 0.9825
# epoch 18 | Train loss 0.008, accuracy 0.9976 | Test loss 0.094, accuracy 0.9834
# epoch 19 | Train loss 0.006, accuracy 0.9982 | Test loss 0.101, accuracy 0.9816
# epoch 20 | Train loss 0.003, accuracy 0.9992 | Test loss 0.098, accuracy 0.9822
# epoch 21 | Train loss 0.001, accuracy 0.9998 | Test loss 0.092, accuracy 0.9850
# epoch 22 | Train loss 0.000, accuracy 0.9999 | Test loss 0.093, accuracy 0.9847
# epoch 23 | Train loss 0.000, accuracy 1.0000 | Test loss 0.094, accuracy 0.9849
# epoch 24 | Train loss 0.000, accuracy 1.0000 | Test loss 0.095, accuracy 0.9851
# epoch 25 | Train loss 0.000, accuracy 1.0000 | Test loss 0.096, accuracy 0.9849
# epoch 26 | Train loss 0.000, accuracy 1.0000 | Test loss 0.097, accuracy 0.9848
# epoch 27 | Train loss 0.000, accuracy 1.0000 | Test loss 0.097, accuracy 0.9848
# epoch 28 | Train loss 0.000, accuracy 1.0000 | Test loss 0.098, accuracy 0.9849
# epoch 29 | Train loss 0.000, accuracy 1.0000 | Test loss 0.098, accuracy 0.9849
# epoch 30 | Train loss 0.000, accuracy 1.0000 | Test loss 0.099, accuracy 0.9849
# epoch 31 | Train loss 0.000, accuracy 1.0000 | Test loss 0.099, accuracy 0.9850
# epoch 32 | Train loss 0.000, accuracy 1.0000 | Test loss 0.100, accuracy 0.9850
# epoch 33 | Train loss 0.000, accuracy 1.0000 | Test loss 0.100, accuracy 0.9850
# epoch 34 | Train loss 0.000, accuracy 1.0000 | Test loss 0.100, accuracy 0.9850
# epoch 35 | Train loss 0.000, accuracy 1.0000 | Test loss 0.101, accuracy 0.9852
# epoch 36 | Train loss 0.000, accuracy 1.0000 | Test loss 0.101, accuracy 0.9853
# epoch 37 | Train loss 0.000, accuracy 1.0000 | Test loss 0.101, accuracy 0.9853
# epoch 38 | Train loss 0.000, accuracy 1.0000 | Test loss 0.102, accuracy 0.9853
# epoch 39 | Train loss 0.000, accuracy 1.0000 | Test loss 0.102, accuracy 0.9853
# epoch 40 | Train loss 0.000, accuracy 1.0000 | Test loss 0.102, accuracy 0.9853
# epoch 41 | Train loss 0.000, accuracy 1.0000 | Test loss 0.103, accuracy 0.9853
# epoch 42 | Train loss 0.000, accuracy 1.0000 | Test loss 0.103, accuracy 0.9853
# epoch 43 | Train loss 0.000, accuracy 1.0000 | Test loss 0.103, accuracy 0.9853
# epoch 44 | Train loss 0.000, accuracy 1.0000 | Test loss 0.103, accuracy 0.9853
# epoch 45 | Train loss 0.000, accuracy 1.0000 | Test loss 0.104, accuracy 0.9853
# epoch 46 | Train loss 0.000, accuracy 1.0000 | Test loss 0.104, accuracy 0.9853
# epoch 47 | Train loss 0.000, accuracy 1.0000 | Test loss 0.104, accuracy 0.9853
# epoch 48 | Train loss 0.000, accuracy 1.0000 | Test loss 0.104, accuracy 0.9853
# epoch 49 | Train loss 0.000, accuracy 1.0000 | Test loss 0.104, accuracy 0.9853

# model = MLP([Linear(784, 1000, ReLU,0),
#              Linear(1000, 500, ReLU,1),
#              Linear(500, 200, ReLU,2),
#              Linear(200, 10, Softmax,3)])
#
# n_epoch = 50
# batchsize = 100
# lr = 0.7