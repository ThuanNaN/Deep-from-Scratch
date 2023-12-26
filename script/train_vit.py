import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from dataset_loader.cifar10 import CIFAR10_Loader
from code_fs.models import ViT
from training_func import ClassificationTraining
from utils import get_device

cifar10_loader = CIFAR10_Loader()
dataloaders = cifar10_loader.get_dataloader(batch_size=256)


model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = ClassificationTraining(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.CrossEntropyLoss(),
    dataloaders=dataloaders,
    epochs=1
)


trainer.train_model(device=get_device())

