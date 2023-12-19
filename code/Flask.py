# from Animal import initialize_model
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
app = Flask(__name__)
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

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = initialize_model(num_classes=37)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
print("Current working directory:", os.getcwd())
@app.route('/upload', methods=['POST'])


def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # 转换图像为模型输入
        image = Image.open(filepath)
        image = test_transform(image).unsqueeze(0)

        # 进行预测
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted.item()  # 假设是分类任务

        return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
