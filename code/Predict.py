from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

def initialize_model(num_classes):
    model = models.resnet101(pretrained=True)

    # 解冻部分层
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

device = torch.device("mps")
model = initialize_model(num_classes=37).to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_path):
    # 加载图片
    image = Image.open(image_path)

    # 应用测试时的转换
    image = test_transform(image)
    
    # 添加批次维度
    image = image.unsqueeze(0)
    return image

def predict_image(image_path):
    image = process_image(image_path)
    
    # 将图片数据移到模型所在的设备上
    image = image.to(device)

    # 预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                image, label = super(CustomImageFolder, self).__getitem__(index)
                return image, label
            except UnidentifiedImageError:
                # 如果遇到损坏的图像，则跳过该图像
                print(f"跳过索引 {index} 处的损坏图片。")
                index = (index + 1) % len(self)

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(128, padding=4),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = ' '  # 替换为实际图片路径
prediction = predict_image(image_path)
train_data = CustomImageFolder(root='data/train/', transform=train_transform)
class_to_idx = train_data.class_to_idx
print(class_to_idx)  # 打印类别名称和对应的索引

print('Predicted class:', prediction)

