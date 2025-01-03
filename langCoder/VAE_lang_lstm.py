import torch
import torch.nn as nn
from torch.nn import functional as F
from base_model import gumbel_softmax, straight_through_discretize
from langCoder.VAE_lang_gru import init_weights


class VAE(nn.Module):
    def __init__(self, input_size, word_length, dictionary_size, device, temperature=1.0):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.word_length = word_length
        self.dictionary_size = dictionary_size
        self.device = device
        self.temperature = temperature

        self.encoder_lstm = nn.LSTMCell(self.input_size, self.input_size)
        self.hidden_to_token = nn.Linear(self.input_size, self.dictionary_size)
        self.token_to_hidden = nn.Linear(self.dictionary_size, self.input_size)
        self.decoder_lstm = nn.LSTMCell(self.input_size, self.input_size)
        self.output_mean = nn.Linear(self.input_size, self.input_size)
        self.output_logvar = nn.Linear(self.input_size, self.input_size)
        self.apply(init_weights)

    def Encoder(self, x, sampling=True):
        samples, logits, messages = [], [], []
        batch_size = x.shape[0]

        hx = torch.zeros(batch_size, self.input_size, device=self.device)
        cx = x
        lstm_input = torch.zeros(batch_size, self.input_size, device=self.device)

        for num in range(self.word_length):
            hx, cx = self.encoder_lstm(lstm_input, (hx, cx))
            pre_logits = self.hidden_to_token(hx)
            logits.append(pre_logits)

            if sampling and self.training:
                z_sampled_soft = gumbel_softmax(pre_logits, self.temperature)
            else:
                z_sampled_soft = torch.softmax(pre_logits, dim=-1)

            z_sampled_onehot, word = straight_through_discretize(z_sampled_soft)
            samples.append(z_sampled_onehot)
            messages.append(word)
            lstm_input = self.token_to_hidden(z_sampled_onehot)

        logits = torch.stack(logits).permute(1, 0, 2)
        samples = torch.stack(samples).permute(1, 0, 2)
        messages = torch.stack(messages).t()
        return samples, logits, messages

    def Decoder(self, z):
        batch_size = z.shape[0]
        z_embeddings = self.token_to_hidden(z.contiguous().view(-1, z.shape[-1])).view(batch_size, self.word_length, -1)
        hx = torch.zeros(batch_size, self.input_size, device=self.device)
        cx = torch.zeros(batch_size, self.input_size, device=self.device)

        for n in range(self.word_length):
            inputs = z_embeddings[:, n]
            hx, cx = self.decoder_lstm(inputs, (hx, cx))

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
        # logits_dist = torch.distributions.Categorical(logits=logits)
        prior = torch.log(torch.tensor([1 / self.dictionary_size] * self.dictionary_size, dtype=torch.float).repeat(1, self.word_length, 1)).to(self.device)
        prior_batch = prior.expand(logits.shape)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_batch)
        # prior_dist = torch.distributions.Categorical(logits=prior_batch)
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        return kl.sum(1).sum(0)
