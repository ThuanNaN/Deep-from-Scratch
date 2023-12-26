import logging
from tqdm import tqdm
from utils import colorstr
import torch

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
                
                for inputs, labels in _phase:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

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





