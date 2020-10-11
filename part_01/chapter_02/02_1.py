# import relevant libraries
from torchvision import models

# printing list of all models located in torchvision.models
print(dir(models))

# creating instances of two modes
alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

# printing the architecture of resnet neural network
print(resnet)