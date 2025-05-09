import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import Generator, Discriminator
from dataloader import get_mnist_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
batch_size = 128
lr = 0.0002
num_epochs = 50 # 10
image_size = 28

loader = get_mnist_loader(batch_size)

G = Generator(latent_dim, image_size).to(device)
D = Discriminator(image_size).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

G_losses, D_losses = [], []

for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(loader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        real = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z).detach()

        real_loss = criterion(D(real_imgs), real)
        fake_loss = criterion(D(fake_imgs), fake)
        D_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)
        G_loss = criterion(D(fake_imgs), real)

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")
    G_losses.append(G_loss.item())
    D_losses.append(D_loss.item())

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Training Losses")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")

# Save model
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")
