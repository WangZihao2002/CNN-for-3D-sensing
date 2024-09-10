import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Sequential, Conv1d, MaxPool1d, ReLU, Flatten, Linear
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


data = np.loadtxt('data/train.csv', delimiter=',')
# print(train_data)
X = data[:, :8]
Y = data[:, 8:]
# print(X)
# print(Y)
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)
scaler_Y = MinMaxScaler()
Y = scaler_Y.fit_transform(Y)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
train_rate = 0.7
val_rate = 0.1
num_data = len(X)
num_train = int(num_data * train_rate)
num_val = int(num_data * val_rate)
train_data = X[:num_train]
# print(train_data)
train_labels = Y[:num_train]
# print(train_labels)
val_data = X[num_train:num_train+num_val]
# print(value_data)
val_labels = Y[num_train:num_train+num_val]
# print(value_labels)
test_data = X[num_train+num_val:]
# print(test_data)
test_labels = Y[num_train+num_val:]
# print(test_labels)
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv1d(1, 32, 3, padding=2)
        self.maxpool1 = MaxPool1d(2)
        self.relu1 = ReLU()
        self.conv2 = Conv1d(32, 32, 3, padding=2)
        self.maxpool2 = MaxPool1d(2)
        self.relu2 = ReLU()
        self.conv3 = Conv1d(32, 64, 3, padding=2)
        self.maxpool3 = MaxPool1d(2)
        self.relu3 = ReLU()
        self.flatten = Flatten()
        self.linear1 = Linear(128, 24)
        self.linear2 = Linear(24, 3)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

cnn = CNN()
# print(cnn)
# input = torch.ones((64, 1, 11))
# output = cnn(input)
# print(output.shape)

loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    cnn.train()
    train_loss = 0.0
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn(data)
        result_loss = loss(outputs, labels)
        result_loss.backward()
        optimizer.step()
        train_loss += result_loss.item() * data.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    cnn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = cnn(data)
            result_loss = loss(outputs, labels)
            val_loss += result_loss.item() * data.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Value Loss: {val_loss:.4f}")

# 在测试集上评估模型
cnn.eval()
test_preds = []
test_labels = []
zongshu  = 0
zhengque = 0
with torch.no_grad():
    for data, labels in test_loader:
        outputs = cnn(data)
        test_preds.extend(outputs.numpy())
        test_labels.extend(labels.numpy())

        outputs_t = torch.round(outputs * 9)
        labels_t = torch.round(labels * 9)
        matches = torch.all(outputs_t == labels_t, dim=1)
        correct_count = torch.sum(matches).item()
        zhengque = zhengque + correct_count
        zongshu = zongshu + outputs_t.size(0)

# 计算准确率
accuracy = zhengque / zongshu
# 打印准确率
print(f'Accuracy: {accuracy * 100:.2f}%')
test_preds = scaler_Y.inverse_transform(np.array(test_preds))
test_labels = scaler_Y.inverse_transform(np.array(test_labels))

# test_labels 和 test_preds 是多目标的，并且有三列（即三个标签）
num_targets = 3
plt.figure(figsize=(12, 8))
for i in range(num_targets):
    plt.subplot(num_targets, 1, i + 1)
    plt.plot(test_labels[:, i], label=f'True Values (Target {i+1})')
    plt.plot(test_preds[:, i], label=f'Predicted Values (Target {i+1})')
    plt.xlabel('Sample')
    plt.ylabel(f'Target {i+1}')
    plt.title(f'True vs. Predicted Values (Target {i+1})')
    plt.legend()
plt.tight_layout()
plt.show()