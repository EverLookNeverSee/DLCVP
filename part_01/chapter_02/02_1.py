# import relevant libraries
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

# printing list of all models located in torchvision.models
print(dir(models))

# creating instances of two modes
alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

# printing the architecture of resnet neural network
print(resnet)

# preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
