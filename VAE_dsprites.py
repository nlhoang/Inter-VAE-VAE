import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(4096, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc_mu = nn.Linear(100, latent_dim)
        self.fc_logvar = nn.Linear(100, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(x.size(0), -1)  # Flatten the features
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 500)
        self.fc5 = nn.Linear(500, 4096)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc5(x))
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
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def elbo(recon_x, x, mu, logvar, beta=1):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return torch.mean(recon_loss + beta * KLD), torch.mean(recon_loss), torch.mean(KLD)


def train(model, dataloader, learning_rate, device, epochs=100, beta=1, saved='VAE_dsprites.pth'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.train()
    D = len(dataloader.dataset)
    for epoch in range(epochs):
        train_loss, rec_loss, kl_loss = 0, 0, 0
        for batch_idx, data in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, rec, kld = elbo(recon, data, mu, logvar, beta)
            train_loss += loss.item()
            rec_loss += rec.item()
            kl_loss += kld.item()
            loss.backward()
            optimizer.step()
        print('====> Epoch: {}, Recon: {:.4f}, KLD: {:.4f}'.format(epoch, train_loss / D, rec_loss / D, kl_loss / D))
    torch.save(model.state_dict(), saved)


def get_latents(model, dataloader, device):
    model.eval()
    means = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encoder(data)
            means.append(mu.cpu().numpy())
    means = np.concatenate(means, axis=0)
    return means
