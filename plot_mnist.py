import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Load dataset
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=16, shuffle=True)

# Get one batch
images, labels = next(iter(loader))

# Display images with their labels
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("MNIST Samples with True Labels", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        # Display image
        img = images[i].permute(1, 2, 0)  # CxHxW -> HxWxC
        ax.imshow(img)
        # Display true label
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    else:
        # Hide unused subplots
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()
