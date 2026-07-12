import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split


transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

# load the training dataset
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Dataset statistics: ")

print(f"Training images: {len(full_trainset)}")
print(f"Test images: {len(testset)}")
print(f"Number of classes: {len(classes)}")

image, label = trainset[0]

print(f"Image shape: {image.shape}")
# image size: 32*32 pizels; color channels: RGB (3 channels)
print(f"Label: {classes[label]}")

# Each class contains 5,000 training images and 1,000 test images, making the dataset balanced

# show an img

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")

# get random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images

grid = torchvision.utils.make_grid(images, nrow=8)

plt.figure(figsize=(8,8))
imshow(grid)
plt.title("Random CIFAR-10 Training Images")
plt.show()

# print labels
print("  ".join(classes[label] for label in labels))

# class distribution
from collections import Counter

label_counts = Counter(full_trainset.targets)

for i, class_name in enumerate(classes):
    print(f"{class_name:10s}: {label_counts[i]}")

# visualize
plt.figure(figsize=(4,2))
plt.bar(classes, [label_counts[i] for i in range(10)])
plt.title("Training Images per Class")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.show()

# create the validation split
# valset used to monitor generalization and detect overfitting

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(
    full_trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training images (80%): {len(trainset)}"
      f"\nValidation images (20%): {len(valset)}"
      f"\nTest images: {len(testset)}")