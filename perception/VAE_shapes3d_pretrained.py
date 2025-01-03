import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import param_count, save_toFile, Shapes3DDataset


class EncImg(nn.Module):
    def __init__(self, latent_dim):
        super(EncImg, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_latent = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_latent(x)
        return x


class DecImg(nn.Module):
    def __init__(self, latent_dim):
        super(DecImg, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: 32x16x16
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Output: 16x32x32
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)  # Output: 3x64x64

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 8, 8)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        recon = torch.sigmoid(self.deconv3(z))
        return recon


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.backbone = EncImg(256)
        saved_model = 'pretrained/shapes3d_encAE_d256.pth'
        self.backbone.load_state_dict(torch.load(saved_model))
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256)
        self.backbone = DecImg(256)
        saved_model = 'pretrained/shapes3d_decAE_d256.pth'
        self.backbone.load_state_dict(torch.load(saved_model))
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.backbone(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        recon = self.decoder(latent)
        #print(f"mu: {mu.mean()}, logvar: {logvar.mean()}, latent: {latent.mean()}, recon: {recon.mean()}")
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def elbo(recon_x, x, mu, logvar, beta=1, use_mse=True, resnet=False):
    if resnet:
        recon_x = recon_x.view(-1, 3 * 224 * 224)
        x = x.view(-1, 3 * 224 * 224)
    else:
        recon_x = recon_x.view(-1, 3 * 64 * 64)
        x = x.view(-1, 3 * 64 * 64)

    if use_mse:
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    else:
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return torch.mean(reconstruction_loss + beta * KLD), torch.mean(reconstruction_loss), torch.mean(KLD)


def train(model, dataloader, learning_rate, device, epochs=100, beta=1, saved=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.train()
    D = len(dataloader.dataset)
    for epoch in range(epochs):
        train_loss, rec_loss, kl_loss = 0, 0, 0
        for batch_idx, data in enumerate(dataloader):
            data = data.float().to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, rec, kld = elbo(recon, data, mu, logvar, beta)
            train_loss += loss.item()
            rec_loss += rec.item()
            kl_loss += kld.item()
            loss.backward()
            optimizer.step()
        print('====> Epoch {} - Loss: {:.4f}, Recon: {:.4f}, KL: {:.4f}'.format(epoch, train_loss / D, rec_loss / D, kl_loss / D))
    if saved is not None:
        torch.save(model.state_dict(), saved)


def get_latents(model, dataloader, device):
    model.eval()
    means = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.float().to(device)
            mu, _ = model.encoder(data)
            means.append(mu.cpu().numpy())
    means = np.concatenate(means, axis=0)
    return means


def display_reconstruction(model, dataset, num_images, device):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    original_images = next(iter(data_loader))
    original_images = original_images.float().to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(original_images)
        z = model.reparameterize(mu, logvar)  # Sampling from the latent space
        reconstructed_images = model.decoder(z)

    original_images = original_images.cpu().numpy()
    reconstructed_images = reconstructed_images.cpu().numpy()
    original_images = original_images.transpose(0, 2, 3, 1)
    reconstructed_images = reconstructed_images.transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    IS_SERVER = False
    path = '/data/' if IS_SERVER else '../data/'
    epochs = 10

    batch_size = 64
    lr = 1e-3
    latent_dim = 50
    data_format = 3 * 64 * 64
    data_size = 32000
    beta = 1

    data_link = path + 'shapes3d/images_00.npy'
    model_saved = 'VAE_shapes3d_d50.pth'
    data_saved = 'latents_shapes3d_VAE_d50.npy'

    dataset = Shapes3DDataset([data_link])
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = VAE(latent_dim=latent_dim).to(device)
    print('Model Size: {}'.format(param_count(model)))
    train(model, dataloader_train, lr, device, epochs, beta=beta, saved=model_saved)
    display_reconstruction(model, dataset, num_images=5, device=device)
    latents = get_latents(model, dataloader_test, device)
    save_toFile('../pretrained/', data_saved, latents, rows=1)
