import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import utils
from  torch import  tensor
from nets.pspnet import PSPNet
# 定义预处理函数
def visualize_attention_map(image, attention_map):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(attention_map, alpha=0.9, cmap='Blues')
    plt.show()


preprocess = transforms.Compose([
    transforms.Resize((473, 473)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
input_shape = [473, 473]
num_classes = 3
backbone = 'mobilenet'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PSPNet(num_classes=num_classes, backbone=backbone, downsample_factor=16, aux_branch=False, pretrained=False).to(
    device)
model.load_state_dict(torch.load('./model_data/best_epoch_weights前四加后一cbam.pth', map_location=torch.device('cuda')))
model.eval()

# 加载图片
img_path = './76.jpg'
img = Image.open(img_path)

# 预处理图片并将其扩展一个维度以适应模型输入
img = preprocess(img).to(device)
img = img.unsqueeze(0)

# 将图片输入模型，获取输出
with torch.no_grad():
    attention_map = model(img)
    output=model(img)
# 获取注意力图，将其转为 numpy 数组


attention_map = attention_map.detach().cpu().numpy()[0]


# 将图片和注意力图可视化
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(np.transpose(img.cpu().squeeze().numpy(), (1, 2, 0)))

ax[1].imshow(visualize_attention_map(np.transpose(img.cpu().squeeze().numpy(), (1, 2, 0)), np.transpose(attention_map, (1, 2, 0))))


plt.show()
