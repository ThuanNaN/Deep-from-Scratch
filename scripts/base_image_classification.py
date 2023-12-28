import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import logging
from tqdm import tqdm
from utils import colorstr, get_device
import torch
from datasets import load_dataset
from torchvision.transforms import (
    Compose, 
    RandomResizedCrop, 
    RandomHorizontalFlip, 
    ToTensor, 
    Resize, 
    CenterCrop, 
    Normalize)
from torch.utils.data import DataLoader
from code_fs.models import ViT

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Classification-Training")


class ClassificationTraining:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 dataloaders: dict,
                 epochs: int = 100,
                 ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.epochs = epochs
        
        self.hist = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_model(self, device: str = 'cuda'):
        self.model.to(device)
        
        best_val_acc = 0.0
        for epoch in range(self.epochs):
            LOGGER.info(colorstr(f'\nEpoch {epoch}/{self.epochs-1}:'))
            for phase in ['train', 'val']:
                if phase == 'train':
                    LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                                        ('Training:', 'gpu_mem', 'loss', 'acc'))
                    self.model.train()
                else:
                    LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                                        ('Validation:', 'gpu_mem', 'loss', 'acc'))
                    self.model.eval()
                
                running_items = 0
                running_loss = 0.0
                running_corrects = 0
                _phase = tqdm(self.dataloaders[phase],
                              total=len(self.dataloaders[phase]),
                              bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                              unit='batch')
                
                for batch in _phase:
                    inputs = batch["inputs"].to(device)
                    labels = batch["labels"].to(device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    running_items += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / running_items
                    epoch_acc = running_corrects / running_items
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                    desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss /
                                                    running_items, running_corrects / running_items)
                    _phase.set_description_str(desc)

            if phase == 'train':
                self.hist["train_loss"].append(epoch_loss)
                self.hist["train_acc"].append(epoch_acc.item())
            
            else:
                self.hist["val_loss"].append(epoch_loss)
                self.hist["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc

        LOGGER.info(f"Best val_acc: {best_val_acc:.6f}")


def main():
    train_ds = load_dataset("cifar10", split="train")
    val_ds = load_dataset("cifar10", split="test")

    size = (224, 224)
    normalize = Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize, 
    ])
    val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["img"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["img"]]
        return example_batch

    train_dataset = train_ds.with_transform(preprocess_train)
    eval_dataset = val_ds.with_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"inputs": pixel_values, "labels": labels}

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=8)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=8)


    dataloaders = {
        "train": train_dataloader,
        "val": eval_dataloader
    }

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = ClassificationTraining(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloaders=dataloaders,
        epochs=1
    )

    trainer.train_model(device=get_device())




main()



