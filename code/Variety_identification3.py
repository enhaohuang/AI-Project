import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

def initialize_model(num_classes):
    # 加载预训练的 ResNet101 模型
    model = models.resnet101(pretrained=True)

    # 解冻特定的层以便在训练过程中更新它们的权重
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 获取模型全连接层的输入特征数，并替换为新的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# 定义训练数据的转换
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),              # 将图像大小调整为128x128
    transforms.RandomHorizontalFlip(),          # 随机水平翻转图像
    transforms.RandomRotation(10),              # 随机旋转图像最多10度
    transforms.RandomCrop(128, padding=4),      # 随机裁剪图像
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # 随机调整图像的亮度和对比度
    transforms.ToTensor(),                      # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化图像
])

# 定义测试数据的转换
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),              # 将图像大小调整为128x128
    transforms.ToTensor(),                      # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化图像
])

# 自定义图像数据集类，处理损坏的图像
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                # 尝试获取图像及其标签，如果成功则返回
                image, label = super(CustomImageFolder, self).__getitem__(index)
                return image, label
            except UnidentifiedImageError:
                # 如果遇到损坏的图像，则跳过该图像
                print(f"跳过索引 {index} 处的损坏图片。")
                index = (index + 1) % len(self)

# 创建训练和测试数据集
train_data = CustomImageFolder(root='data/train/', transform=train_transform)
test_data = CustomImageFolder(root='data/test/', transform=test_transform)

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 设置计算设备（如GPU）
device = torch.device("mps")

# 初始化模型并将其转移到计算设备上
model = initialize_model(num_classes=37).to(device)

# 加载预先保存的模型权重
model.load_state_dict(torch.load('model_weights.pth', map_location=device))

# 设置模型为评估模式
model.eval()

# 使用不同的学习率为不同的模型层定义优化器
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},   # 为全连接层使用较高的学习率
    {'params': model.layer4.parameters(), 'lr': 1e-4} # 为layer4层使用较低的学习率
], lr=1e-5)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# 设置训练的周期数
num_epochs = 30

# 初始化存储训练和验证损失和准确率的列表
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 训练循环
for epoch in range(num_epochs):
    # 设置模型为训练模式
    model.train()
    
    # 初始化累计变量
    running_loss = 0.0
    correct = 0
    total = 0

    # 遍历训练数据
    for i, (images, labels) in enumerate(train_loader):
        # 将数据传输到计算设备上
        images, labels = images.to(device), labels.to(device)

        # 优化器梯度归零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算并存储平均训练损失和准确率
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # 验证模型
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_losses.append(val_running_loss / len(test_loader))
    val_accuracies.append(100 * correct / total)

    # 打印每个周期的训练和验证结果
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%, Val Accuracy: {val_accuracies[-1]:.2f}%')

# 绘制训练和验证损失以及准确率的比较图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# 保存训练参数
torch.save(model.state_dict(), 'model_weights.pth')
