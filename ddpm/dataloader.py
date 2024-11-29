import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_transformed_dataset(img_size=32, batch_size=128) -> DataLoader:
    # Load dataset and perform data transformations
    train_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scale data into [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Scale between [-1, 1]
    ])
    test_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root="./datasets", 
                                                train=True,
                                                download=False,
                                                transform=train_data_transform)
    
    test_dataset = torchvision.datasets.CIFAR10(root="./datasets",
                                               train=False, 
                                               download=False,
                                               transform=test_data_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           drop_last=True)
    
    return train_loader, test_loader


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # If image is a batch, take the first image
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


if __name__ == "__main__":
    train_loader, test_loader = load_transformed_dataset()
    image, _ = next(iter(train_loader))
    plt.imshow(show_tensor_image(image))
    plt.show()
