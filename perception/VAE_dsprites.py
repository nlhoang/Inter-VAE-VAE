import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import param_count, save_toFile, DsrpitesDataset


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


def train(model, dataloader, learning_rate, device, epochs=100, beta=1, saved=None):
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
    if saved is not None:
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


def display_reconstruction(model, dataset, num_images, device):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    original_images = next(iter(data_loader))
    original_images = original_images.view(original_images.size(0), -1).to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(original_images)
        z = model.reparameterize(mu, logvar)  # Sampling from the latent space
        reconstructed_images = model.decoder(z)

    original_images = original_images.view(original_images.size(0), 1, 64, 64)
    reconstructed_images = reconstructed_images.view(reconstructed_images.size(0), 1, 64, 64)
    original_images = original_images.cpu().numpy()
    reconstructed_images = reconstructed_images.cpu().numpy()

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
    epochs = 50

    batch_size = 64
    lr = 1e-3
    latent_dim = 10
    data_format = 64 * 64
    data_size = 18432
    beta = 1
    data_link = path + 'dsprites/images_00.npy'
    model_saved = 'VAE_dsprites_lin_d50.pth'
    data_saved = 'latents_dsprites_test_vae10.npy'

    dataset = DsrpitesDataset([data_link])
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = VAE(latent_dim=latent_dim).to(device)
    print('Model Size: {}'.format(param_count(model)))
    train(model, dataloader_train, lr, device, epochs, beta=beta, saved=model_saved)
    display_reconstruction(model, dataset, num_images=5, device=device)
    latents = get_latents(model, dataloader_test, device)
    save_toFile('../pretrained/', data_saved, latents, rows=1)
