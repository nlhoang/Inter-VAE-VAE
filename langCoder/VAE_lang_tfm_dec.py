import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import gumbel_softmax, straight_through_discretize


class VAE(nn.Module):
    def __init__(self, device, input_size, dictionary_size=100, word_length=10, hidden_size=200, nhead=5,
                 num_layers=2, dim_feedforward=2048, temperature=1.0):
        super().__init__()
        self.input_size = input_size
        self.dictionary_size = dictionary_size
        self.word_length = word_length
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.device = device

        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.memory_fc = nn.Linear(hidden_size, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=0.1, activation="relu", batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_classifier = nn.Linear(hidden_size, dictionary_size)
        self.target_embeddings = nn.Embedding(word_length, hidden_size)

        self.decoder_fc1 = nn.Linear(word_length * dictionary_size, hidden_size * 3)
        self.decoder_fc2 = nn.Linear(hidden_size * 3, hidden_size)
        self.output_mean = nn.Linear(hidden_size, input_size)
        self.output_logvar = nn.Linear(hidden_size, input_size)

    def Encoder(self, x, sampling=True):
        emb = F.relu(self.input_embedding(x))
        memory = F.relu(self.memory_fc(emb))
        memory = memory.unsqueeze(1)  # [batch_size, 1, hidden_size]

        target_positions = torch.arange(self.word_length, device=x.device)
        target_emb = self.target_embeddings(target_positions)
        target_emb = target_emb.unsqueeze(0).repeat(x.size(0), 1, 1)

        decoded_seq = self.transformer_decoder(tgt=target_emb, memory=memory)
        logits = self.token_classifier(decoded_seq)  # [batch_size, word_length, dictionary_size]

        if sampling and self.training:
            z_sampled_soft = gumbel_softmax(logits, temperature=self.temperature)
        else:
            z_sampled_soft = F.softmax(logits, dim=-1)

        one_hot_tokens, messages = straight_through_discretize(z_sampled_soft)
        return one_hot_tokens, logits, messages

    def Decoder(self, z):
        z = z.to(self.device)
        decoded = z.view(-1, self.word_length * self.dictionary_size)
        decoded = F.relu(self.decoder_fc1(decoded))
        decoded = F.relu(self.decoder_fc2(decoded))
        recon_mean = self.output_mean(decoded)
        recon_logvar = self.output_logvar(decoded)
        recon = self.reparameterize(recon_mean, recon_logvar)
        return recon, recon_mean, recon_logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, sampling=True):
        one_hot_tokens, logits, messages = self.Encoder(x, sampling=sampling)
        recons, _, _ = self.Decoder(one_hot_tokens)
        return recons, one_hot_tokens, logits, messages

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
        prior = torch.log(torch.tensor([1.0 / self.dictionary_size] * self.dictionary_size, device=self.device))
        prior_dist = torch.distributions.OneHotCategorical(logits=prior.expand_as(logits))
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        return kl.sum(1).mean()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils import param_count, save_toFile
    from VAE_lang_gru import train, get_messages

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    epochs = 2
    batch_size = 64
    hidden_size = 200
    lr = 1e-3
    word_length = 10
    dictionary_size = 100
    input_size = 50
    data_link = '../pretrained/latents_dsprites_VAE_lin_d50_a.npy'
    data_saved = 'messages_dsprites_test.npy'
    dataset = np.genfromtxt(data_link, delimiter=',', dtype=float)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset, batch_size=18432, shuffle=False)
    model = VAE(input_size=input_size, dictionary_size=dictionary_size, word_length=word_length,
                hidden_size=hidden_size, nhead=4, num_layers=2, temperature=0.5, device=device).to(device)
    print('Model Size: {}'.format(param_count(model)))
    train(model=model, dataloader=dataloader_train, learning_rate=lr, device=device, epochs=epochs)
    _, messages = get_messages(model, dataloader_test, device)
    print(messages)
    save_toFile('experiments/', data_saved, messages, rows=1)
