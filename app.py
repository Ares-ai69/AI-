from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# 加载AI预训练模型（前沿：ResNet图像分类）
model = models.resnet18(pretrained=True)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# 你的信息
STUDENT_ID = "你的学号"
NAME = "你的姓名"

@app.route('/')
def index():
    return render_template("index.html", sid=STUDENT_ID, name=NAME)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['img']
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    return f"预测类别索引：{out.argmax().item()}"

if __name__ == '__main__':
    app.run(debug=True)
