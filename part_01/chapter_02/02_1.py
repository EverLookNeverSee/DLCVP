import json
from torchvision.datasets.utils import download_url

from torchvision import models
from torchvision import transforms
from PIL import Image
import torch


download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".", "imagenet_class_index.json")


with open("imagenet_class_index.json", "r") as h:
    labels = json.load(h)


alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


download_url("https://farm1.static.flickr.com/152/434505223_8d1890e1e2.jpg", ".", "sample_dog.jpg")
img = Image.open("sample_dog.jpg")
img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()

with torch.no_grad():
    out = resnet(batch_t)


label_indices = out.argmax(dim=1)
for p in label_indices:
    print(labels[str(p.item())])
