import warnings

import cv2
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34, resnet50

from utils import *

# 忽略特定类型的警告
# warnings.filterwarnings("ignore", category=UserWarning)

# 加载训练模型
resnet_model = resnet18()
# 获取全连接层前的输入个数
num_ftrs = resnet_model.fc.in_features
# 修改全连接层的输出神经元个数
resnet_model.fc = nn.Sequential(nn.Linear(num_ftrs, args.num_classes), nn.LogSoftmax(dim=1))

torch.save(resnet_model, "model_structure/resnet18_model.pth")
# 将模型切换到评估模式，并进行预测
resnet_model.eval()
