from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class CIFAR10_Loader:
    def __init__(self, 
                 data_dir: str = "./cache", 
                 mean = (0.4914, 0.4821, 0.4465),
                 std = (0.2471, 0.2435, 0.2616)
                 ) -> None:

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        valid_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
        self.valid_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=valid_transforms)


    def get_dataloader(self, batch_size, num_workers=0):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, num_workers=num_workers)

        return {
            "train": train_loader,
            "val": valid_loader
        }
    

"""
Usage:

cifar10_loader = CIFAR10_Loader()
train_loader, val_loader = cifar10_loader.get_dataloader(batch_size=bs)

"""
