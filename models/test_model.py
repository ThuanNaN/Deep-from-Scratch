import torch
from vgg import vgg16


image_tensor = torch.randn((1,3,224,224))


# <<<<<<__VGG___<<<<<
vgg16_model = vgg16(num_classes=2)
output = vgg16_model(image_tensor)
print(output.size())
# <<<<<<__VGG___<<<<<

