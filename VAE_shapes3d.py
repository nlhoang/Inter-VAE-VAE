import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np


class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super(CNNDecoder, self).__init__()
        self.lin = nn.Linear(latent_dim, 256 * 4 * 4)
        self.fc1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.fc2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc4 = nn.ConvTranspose2d(32, 16, kernel_size=1, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.fc5 = nn.ConvTranspose2d(16, 8, kernel_size=1, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(8)
        self.fc6 = nn.ConvTranspose2d(8, output_channels, kernel_size=2, stride=2, padding=1)

    def forward(self, z):
        z = self.lin(z)
        z = z.view(-1, 256, 4, 4)
        z = F.relu(self.bn1(self.fc1(z)))
        z = F.relu(self.bn2(self.fc2(z)))
        z = F.relu(self.bn3(self.fc3(z)))
        z = F.relu(self.bn4(self.fc4(z)))
        z = F.relu(self.bn5(self.fc5(z)))
        recon = torch.sigmoid(self.fc6(z))
        return recon


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        recon = self.decoder(latent)
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def elbo(recon_x, x, mu, logvar, beta=1, use_mse=True):
    recon_x = recon_x.view(-1, 3 * 224 * 224)
    x = x.view(-1, 3 * 224 * 224)

    if use_mse:
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    else:
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return torch.mean(reconstruction_loss + beta * KLD), torch.mean(reconstruction_loss), torch.mean(KLD)


def train(model, dataloader, learning_rate, device, epochs=100, beta=1, saved='VAE_shapes3d.pth'):
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
            print(loss.item())
        print('====> Epoch {} - Loss: {:.4f}, Recon: {:.4f}, KL: {:.4f}'.format(epoch, train_loss / D, rec_loss / D, kl_loss / D))
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
