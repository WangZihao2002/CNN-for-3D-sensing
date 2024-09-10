import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data = np.loadtxt('data/train.csv', delimiter=',')
# print(train_data)
X = data[:, :8]
y = data[:, 8:]
# 假设数据已经加载到X（输入特征）和y（输出标签）中
# X shape: (samples, 8)
# y shape: (samples, 3)  # 3个输出：x, y, z力的分量

# 数据预处理
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 对输出进行归一化
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# 重塑数据以适应CNN
X_reshaped = np.column_stack([X_scaled, X_scaled[:, 0]])
X_reshaped = X_reshaped.reshape(-1, 1, 9)  # shape: (samples, 1, 9) for PyTorch CNN

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X_reshaped)
y_tensor = torch.FloatTensor(y_scaled)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义模型
class CircularCNN(nn.Module):
    def __init__(self):
        super(CircularCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 4, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 4)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = CircularCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    mae = nn.L1Loss()(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Test MAE: {mae.item():.4f}')

# 计算每个分量的MAE
y_pred_np = y_pred.numpy()
y_test_np = y_test.numpy()

y_pred_original = scaler_y.inverse_transform(y_pred_np)
y_test_original = scaler_y.inverse_transform(y_test_np)

mae_x = np.mean(np.abs(y_pred_original[:, 0] - y_test_original[:, 0]))
mae_y = np.mean(np.abs(y_pred_original[:, 1] - y_test_original[:, 1]))
mae_z = np.mean(np.abs(y_pred_original[:, 2] - y_test_original[:, 2]))

print(f"MAE for x component: {mae_x:.4f}")
print(f"MAE for y component: {mae_y:.4f}")
print(f"MAE for z component: {mae_z:.4f}")

# 计算力的大小MAE
predicted_magnitude = np.linalg.norm(y_pred_original, axis=1)
actual_magnitude = np.linalg.norm(y_test_original, axis=1)
mae_magnitude = np.mean(np.abs(predicted_magnitude - actual_magnitude))
print(f"MAE for force magnitude: {mae_magnitude:.4f}")

# 创建图表
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
components = ['X', 'Y', 'Z']

for i in range(3):
    axs[i].scatter(y_test_original[:, i], y_pred_original[:, i], alpha=0.5)
    axs[i].plot([y_test_original[:, i].min(), y_test_original[:, i].max()],
                [y_test_original[:, i].min(), y_test_original[:, i].max()],
                'r--', lw=2)
    axs[i].set_xlabel(f'True {components[i]} Force')
    axs[i].set_ylabel(f'Predicted {components[i]} Force')
    axs[i].set_title(f'{components[i]} Component')

plt.tight_layout()
plt.show()

# 计算并打印R^2分数
from sklearn.metrics import r2_score

r2_scores = [r2_score(y_test_original[:, i], y_pred_original[:, i]) for i in range(3)]

for i, score in enumerate(r2_scores):
    print(f"R^2 score for {components[i]} component: {score:.4f}")

# 绘制力的大小对比图
true_magnitude = np.linalg.norm(y_test_original, axis=1)
pred_magnitude = np.linalg.norm(y_pred_original, axis=1)

plt.figure(figsize=(8, 6))
plt.scatter(true_magnitude, pred_magnitude, alpha=0.5)
plt.plot([true_magnitude.min(), true_magnitude.max()],
         [true_magnitude.min(), true_magnitude.max()],
         'r--', lw=2)
plt.xlabel('True Force Magnitude')
plt.ylabel('Predicted Force Magnitude')
plt.title('Force Magnitude: True vs Predicted')
plt.tight_layout()
plt.show()

# 计算并打印力大小的R^2分数
magnitude_r2 = r2_score(true_magnitude, pred_magnitude)
print(f"R^2 score for force magnitude: {magnitude_r2:.4f}")