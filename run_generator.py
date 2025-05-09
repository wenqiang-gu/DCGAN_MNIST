import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models import Generator # Assuming Generator class is in models.py

# Parameters
latent_dim = 100  # Should match the latent_dim used during training
image_size = 28  # MNIST images are 28x28
generator_path = "generator.pth"
num_images_to_generate = 16 # Number of images to generate in a grid

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator
generator = Generator(latent_dim, image_size).to(device)

# Load the trained generator weights
try:
    generator.load_state_dict(torch.load(generator_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Generator model file not found at {generator_path}")
    print("Please ensure the model is trained and the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading generator model: {e}")
    exit()

# Set the generator to evaluation mode
generator.eval()

# Generate images
with torch.no_grad():
    # Create random noise as input to the generator
    noise = torch.randn(num_images_to_generate, latent_dim, device=device)
    # Generate images
    generated_images = generator(noise).cpu()

# Plot and save the generated images
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title(f"Generated MNIST-like Images (from {generator_path})")
plt.imshow(vutils.make_grid(generated_images, padding=2, normalize=True).permute(1, 2, 0))
output_image_path = "generated_mnist_samples.png"
plt.savefig(output_image_path)
plt.show()

print(f"Generated images saved to {output_image_path}")
