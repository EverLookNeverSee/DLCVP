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

# opening sample image
img = Image.open("sample_dog.jpg")
# passing image through preprocessing pipeline
img_t = preprocess(img)

# reshaping, cropping and normalizing the tensor
batch_t = torch.unsqueeze(img_t, 0)

# put in network in evaluate mode
resnet.eval()

# inference the model
out = resnet(batch_t)
print(out)

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readline()]

_, index = torch.max(out, 1)


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()
