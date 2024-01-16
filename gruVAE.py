import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif type(layer) == nn.LSTMCell:
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data, nonlinearity='tanh')
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                start, end = layer.bias_ih.size(0) // 4, layer.bias_ih.size(0) // 2
                param.data[start:end].fill_(1.)
    elif type(layer) == nn.GRUCell:
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data, nonlinearity='tanh')
            elif 'bias' in name:
                nn.init.zeros_(param.data)


class VAE(nn.Module):
    def __init__(self, input_size, word_length, dictionary_size, device, temperature=1.0):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.word_length = word_length
        self.dictionary_size = dictionary_size
        self.device = device
        self.temperature = temperature

        self.encoder_gru = nn.GRUCell(self.input_size, self.input_size)
        self.hidden_to_token = nn.Linear(self.input_size, self.dictionary_size)
        self.token_to_hidden = nn.Linear(self.dictionary_size, self.input_size)
        self.decoder_gru = nn.GRUCell(self.input_size, self.input_size)
        self.output_mean = nn.Linear(self.input_size, self.input_size)
        self.output_logvar = nn.Linear(self.input_size, self.input_size)
        self.apply(init_weights)

    def Encoder(self, x, sampling=True):
        logits, messages, one_hot_tokens = [], [], []
        batch_size = x.shape[0]

        hx = x
        gru_input = torch.zeros(batch_size, self.input_size, device=self.device)
        for num in range(self.word_length):
            hx = self.encoder_gru(gru_input, hx)
            pre_logits = self.hidden_to_token(hx)

            if sampling and self.training:
                z_sampled_soft = gumble_softmax(pre_logits, self.temperature)
            else:
                z_sampled_soft = torch.softmax(pre_logits, dim=-1)

            logits.append(z_sampled_soft)
            z_sampled_onehot, word = straight_through_discretize(z_sampled_soft)
            one_hot_tokens.append(z_sampled_onehot)
            messages.append(word)
            gru_input = self.token_to_hidden(z_sampled_onehot)

        logits = torch.stack(logits).permute(1, 0, 2)
        one_hot_tokens = torch.stack(one_hot_tokens).permute(1, 0, 2)
        messages = torch.stack(messages).t()
        return one_hot_tokens, logits, messages

    def Decoder(self, z):
        batch_size = z.shape[0]
        z_embeddings = self.token_to_hidden(z.contiguous().view(-1, z.shape[-1])).view(batch_size, self.word_length, -1)
        hx = torch.zeros(batch_size, self.input_size, device=self.device)

        for n in range(self.word_length):
            inputs = z_embeddings[:, n]
            hx = self.decoder_gru(inputs, hx)

        output_mean = self.output_mean(hx)
        output_logvar = self.output_logvar(hx)
        output = self.reparameterize(output_mean, output_logvar)
        return output, output_mean, output_logvar

    def forward(self, input):
        one_hot_tokens, logits, messages = self.Encoder(input)
        recons, _, _ = self.Decoder(one_hot_tokens)
        return recons, one_hot_tokens, logits, messages

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def elbo(self, inputs, recon, logits, beta=1):
        recon_loss = self.compute_recontruct_loss(inputs, recon)
        kld_loss = self.compute_KLD_loss(logits)
        loss = recon_loss + beta * kld_loss
        return loss, recon_loss, kld_loss

    def compute_recontruct_loss(self, inputs, recon, loss='mse'):
        if loss == 'mse':
            recon_loss = F.mse_loss(recon, inputs, reduction='sum') / inputs.size(0)
        else:
            recon_loss = F.binary_cross_entropy_with_logits(recon, inputs, reduction='sum') / inputs.size(0)
        return recon_loss

    def compute_KLD_loss(self, logits):
        logits_dist = torch.distributions.OneHotCategorical(logits=logits)
        prior = torch.log(torch.tensor([1 / self.dictionary_size] * self.dictionary_size, dtype=torch.float).repeat(1, self.word_length, 1)).to(self.device)
        prior_batch = prior.expand(logits.shape)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_batch)
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        return kl.sum(1).sum(0)


def gumble_softmax(logits, temperature=1.0):
    g = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    G = g.sample()
    return F.softmax((logits + G) / temperature, -1)


def straight_through_discretize(z_sampled_soft):
    z_argmax = torch.argmax(z_sampled_soft, dim=-1, keepdim=True)
    z_argmax_one_hot = torch.zeros_like(z_sampled_soft).scatter_(-1, z_argmax, 1)
    z_sampled_onehot_with_grad = z_sampled_soft + (z_argmax_one_hot - z_sampled_soft).detach()
    return z_sampled_onehot_with_grad, z_argmax.squeeze(-1)


def train(model, dataloader, learning_rate, device, epochs=100, saved='gruVAE.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    D = len(dataloader.dataset)
    for epoch in range(epochs):
        train_loss, train_reco, train_KLD = 0, 0, 0
        for batch_idx, data in enumerate(dataloader):
            data = data.float().view(data.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, one_hot_token, logits, mess = model(data)
            loss, recon_loss, kld_loss = model.elbo(data, recon, one_hot_token)
            loss.backward()
            train_loss += loss.item()
            train_reco += recon_loss.item()
            train_KLD += kld_loss.item()
            optimizer.step()
        print('====> Epoch: {}, Recon: {:.4f}, KLD: {:.4f}'.format(epoch, train_reco / D, train_KLD / D))
    torch.save(model.state_dict(), saved)


def get_messages(model, dataloader, device):
    model.eval()
    latent, message = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.float().view(data.size(0), -1).to(device)
            recon, one_hot_token, logits, mes = model(data)
            latent.append(recon.cpu().numpy())
            message.append(mes.cpu().numpy())
    latent = np.concatenate(latent, axis=0)
    message = np.concatenate(message, axis=0)
    return latent, message
