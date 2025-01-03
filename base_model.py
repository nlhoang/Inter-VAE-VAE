import random
import torch
import torch.nn as nn
from torch.nn import functional as F


def compute_similarity_loss(inputs, recon, loss='mse'):
    if loss == 'mse':
        recon_loss = F.mse_loss(recon, inputs, reduction='sum')
    else:
        recon_loss = torch.cosine_similarity(recon, inputs, dim=0)
    return recon_loss


class ReferentialGame:
    # distract = other inputs
    def play_game_input(self, inputs, num_distractors=3):
        batch_size = inputs.size(0)
        one_hot, logits, messages = self.encode(inputs, sampling=True)  # Shape: (batch_size, latent_size) - Encode all inputs to get their messages
        reconstructed_outputs = self.decode(logits)  # Shape: (batch_size, output_size) - Decode each message to get reconstructed output
        all_losses = []  # Initialize list for collecting all losses

        # Iterate over each input in the batch
        for i in range(batch_size):
            true_target = inputs[i]  # Get the true target and its reconstructed output
            true_output = reconstructed_outputs[i]
            true_similarity = compute_similarity_loss(true_target, true_output, loss='mse')

            # Select num_distractors randomly from other items in the batch
            distractor_indices = list(range(batch_size))
            distractor_indices.remove(i)  # Exclude the true target index
            distractor_indices = random.sample(distractor_indices, num_distractors)
            distractors = inputs[distractor_indices]  # Shape: (num_distractors, input_size)
            distractor_similarities = torch.stack([
                compute_similarity_loss(distractor, true_output, loss='mse') for distractor in distractors
            ])

            # Calculate Hinge loss, maximize true similarity while minimizing similarity to distractors
            loss_for_sample = torch.relu(1.0 - true_similarity + distractor_similarities).mean()
            all_losses.append(loss_for_sample)

        loss = torch.stack(all_losses).mean()
        return loss

    # distract = other reconstructed
    def play_game_recon(self, inputs, num_distractors=3):
        batch_size = inputs.size(0)
        one_hot, logits, messages = self.encode(inputs, sampling=True)  # Shape: (batch_size, latent_size) - Encode all inputs to get their messages
        reconstructed_outputs = self.decode(one_hot)  # Shape: (batch_size, output_size) - Decode each message to get reconstructed output
        all_losses = []  # Initialize list for collecting all losses

        # Iterate over each input in the batch
        for i in range(batch_size):
            true_target = inputs[i]  # Get the true target and its reconstructed output
            true_output = reconstructed_outputs[i]
            true_similarity = compute_similarity_loss(true_target, true_output, loss='mse')

            # Select num_distractors randomly from other items in the batch
            distractor_indices = list(range(batch_size))
            distractor_indices.remove(i)  # Exclude the true target index
            distractor_indices = random.sample(distractor_indices, num_distractors)
            distractors = reconstructed_outputs[distractor_indices]  # Shape: (num_distractors, input_size)
            distractor_similarities = torch.stack([
                compute_similarity_loss(distractor, true_target, loss='mse') for distractor in distractors
            ])

            # Calculate Hinge loss, maximize true similarity while minimizing similarity to distractors
            loss_for_sample = torch.relu(1.0 - true_similarity + distractor_similarities).mean()
            all_losses.append(loss_for_sample)

        loss = torch.stack(all_losses).mean()
        return loss


class ReconstructionGame:
    def play_game(self, input, sampling=True):
        one_hot, logits, _ = self.encode(input, sampling)
        recon = self.decode(one_hot)
        loss, recon_loss, kld_loss = self.elbo(input, recon, logits)
        return loss, recon_loss, kld_loss

    def elbo(self, inputs, recon, logits, beta=1):
        recon_loss = self.compute_recontruct_loss(inputs, recon)
        kld_loss = self.compute_KLD_loss(logits)
        loss = recon_loss + beta * kld_loss
        return loss, recon_loss, kld_loss

    def compute_KLD_loss(self, logits):
        logits_dist = torch.distributions.OneHotCategorical(logits=logits)
        prior = torch.log(torch.tensor([1.0 / self.dictionary_size] * self.dictionary_size, device=self.device))
        prior_dist = torch.distributions.OneHotCategorical(logits=prior.expand_as(logits))
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        return kl.sum(1).mean()

    def compute_recontruct_loss(self, inputs, recon, loss='mse'):
        if loss == 'mse':
            recon_loss = F.mse_loss(recon, inputs, reduction='sum') / inputs.size(0)
        else:
            recon_loss = F.binary_cross_entropy_with_logits(recon, inputs, reduction='sum') / inputs.size(0)
        return recon_loss


def gumbel_softmax(logits, temperature=1.0):
    g = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    G = g.sample()
    return F.softmax((logits + G) / temperature, -1)


def straight_through_discretize(z_sampled_soft):
    z_argmax = torch.argmax(z_sampled_soft, dim=-1, keepdim=True)
    z_argmax_one_hot = torch.zeros_like(z_sampled_soft).scatter_(-1, z_argmax, 1)
    z_sampled_onehot_with_grad = z_sampled_soft + (z_argmax_one_hot - z_sampled_soft).detach()
    return z_sampled_onehot_with_grad, z_argmax.squeeze(-1)


class VAE_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, latent_size, dictionary_size, device, temperature=1.0):
        super(VAE_LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dictionary_size = dictionary_size
        self.temperature = temperature
        self.device = device

        if self.input_size is not None:
            self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        if self.output_size is not None:
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        self.encoder_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.hidden_to_token = nn.Linear(self.hidden_size, self.dictionary_size)
        self.token_to_hidden = nn.Linear(self.dictionary_size, self.hidden_size)

    def encode(self, x, sampling=True):
        x = self.input_layer(x)
        one_hot, logits, message = self.encode_variable_length(x, sampling=sampling)
        return one_hot, logits, message

    def decode(self, z):
        output = self.decode_variable_length(z)
        output = self.output_layer(output)
        return output

    def encode_variable_length(self, x, sampling=True):
        _device = x.device
        samples, logits, messages = [], [], []
        batch_size = x.shape[0]
        hx = torch.zeros(batch_size, self.hidden_size, device=_device)
        cx = x
        lstm_input = torch.zeros(batch_size, self.hidden_size, device=_device)

        for num in range(self.latent_size):
            hx, cx = self.encoder_lstm(lstm_input, (hx, cx))
            pre_logits = self.hidden_to_token(hx)  # embedding to catogory logits
            logits.append(pre_logits)

            if sampling and self.training:
                z_sampled_soft = gumbel_softmax(pre_logits, self.temperature)
            else:
                z_sampled_soft = torch.softmax(pre_logits, dim=-1)

            z_sampled_onehot, z_argmax = straight_through_discretize(z_sampled_soft)
            samples.append(z_sampled_onehot)
            messages.append(z_argmax)
            lstm_input = self.token_to_hidden(z_sampled_onehot)

        logits = torch.stack(logits).permute(1, 0, 2)
        samples = torch.stack(samples).permute(1, 0, 2)
        messages = torch.stack(messages).t()
        return samples, logits, messages

    def decode_variable_length(self, z):
        batch_size = z.shape[0]
        _device = z.device

        z_embeddings = self.token_to_hidden(z.contiguous().view(-1, z.shape[-1])).view(batch_size, self.latent_size, -1)  # project one-hot codes into continueious embeddings
        hx = torch.zeros(batch_size, self.hidden_size, device=_device)
        cx = torch.zeros(batch_size, self.hidden_size, device=_device)
        outputs = []
        for n in range(self.latent_size):
            inputs = z_embeddings[:,n]
            hx, cx = self.decoder_lstm(inputs, (hx, cx))
            outputs.append(hx)
        return hx

    def forward(self, input, sampling=True):
        one_hot, logits, messages = self.encode(input, sampling)
        recon = self.decode(one_hot)
        return recon, one_hot, logits, messages


class VAE_LSTM_img(VAE_LSTM):
    def __init__(self, hidden_size, latent_size, dictionary_size, device, temperature=1.0):
        super(VAE_LSTM_img, self).__init__(
            input_size=None,
            output_size=None,
            hidden_size=hidden_size,
            latent_size=latent_size,
            dictionary_size=dictionary_size,
            device=device,
            temperature=temperature,
        )

        self.input_layer = nn.Linear(64 * 8 * 8, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 64 * 8 * 8)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # Output: (3, 64, 64)
            nn.Sigmoid()  # Assuming image pixel values are normalized to [0, 1]
        )

    def encode(self, x, sampling=True):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        one_hot, logits, message = self.encode_variable_length(x, sampling=sampling)
        return one_hot, logits, message

    def decode(self, z):
        output = self.decode_variable_length(z)
        output = self.output_layer(output)
        output = output.view(z.size(0), 64, 8, 8)
        output = self.decoder_conv(output)
        return output


class VAE_TFM(nn.Module):
    def __init__(self, input_size, word_length, dictionary_size, device, temperature=1.0,
                 hidden_size=200, num_heads=5, num_layers=2, dim_feedforward=2048):
        super(VAE_TFM, self).__init__()
        self.input_size = input_size
        self.word_length = word_length
        self.hidden_size = hidden_size
        self.dictionary_size = dictionary_size
        self.device = device
        self.temperature = temperature

        if self.input_size is not None:
            self.encoder_embedding = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, input_size)

        self.encoder = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder, num_layers)
        self.latent_logits = nn.Linear(hidden_size, dictionary_size * word_length)

        self.decoder_fc1 = nn.Linear(word_length * dictionary_size, hidden_size * 3)
        self.decoder_fc2 = nn.Linear(hidden_size * 3, hidden_size)

    def encode(self, x, sampling=True):
        src = self.encoder_embedding(x.to(self.device))
        src = src.unsqueeze(1)
        src = src.permute(1, 0, 2)
        self.encoded = self.encoder_transformer(src)
        logits = self.latent_logits(self.encoded).view(-1, self.word_length, self.dictionary_size)

        if sampling and self.training:
            z_sampled_soft = gumbel_softmax(logits, temperature=self.temperature)
        else:
            z_sampled_soft = torch.softmax(logits, dim=-1)

        one_hot_tokens, messages = straight_through_discretize(z_sampled_soft)
        return one_hot_tokens, logits, messages

    def decode(self, z):
        z = z.to(self.device)
        decoded = z.view(-1, self.word_length * self.dictionary_size)
        decoded = F.relu(self.decoder_fc1(decoded))
        decoded = F.relu(self.decoder_fc2(decoded))
        recon = self.output(decoded)
        return recon

    def forward(self, input):
        one_hot_token, logit, message = self.encode(input)
        recon = self.decode(one_hot_token)
        return recon, one_hot_token, logit, message


class VAE_TFM_img(VAE_TFM):
    def __init__(self, word_length, dictionary_size, device, temperature=1.0,
                 hidden_size=200, num_heads=5, num_layers=2, dim_feedforward=2048):
        super(VAE_TFM_img, self).__init__(
            input_size=None,
            word_length=word_length,
            dictionary_size=dictionary_size,
            device=device,
            temperature=temperature,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

        self.encoder_fc = nn.Linear(64 * 8 * 8, hidden_size)
        self.decoder_fc1 = nn.Linear(word_length * dictionary_size, hidden_size)
        self.decoder_fc2 = nn.Linear(hidden_size, 64 * 8 * 8)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # Output: (3, 64, 64)
            nn.Sigmoid()  # Assuming image pixel values are normalized to [0, 1]
        )

    def encode(self, x, sampling=True):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)
        x = self.encoder_transformer(x)
        logits = self.latent_logits(x).view(-1, self.word_length, self.dictionary_size)

        if sampling and self.training:
            z_sampled_soft = gumbel_softmax(logits, temperature=self.temperature)
        else:
            z_sampled_soft = torch.softmax(logits, dim=-1)

        one_hot_tokens, messages = straight_through_discretize(z_sampled_soft)
        return one_hot_tokens, logits, messages

    def decode(self, z):
        z = z.to(self.device)
        z = z.view(-1, self.word_length * self.dictionary_size)
        z = F.relu(self.decoder_fc1(z))
        z = F.relu(self.decoder_fc2(z))
        z = z.view(z.size(0), 64, 8, 8)
        recon = self.decoder_conv(z)
        return recon


class VAE_TFM2(nn.Module):
    def __init__(self, input_size, word_length, dictionary_size, device, temperature=1.0,
                 hidden_size=200, num_heads=5, num_layers=2, dim_feedforward=2048):
        super(VAE_TFM2, self).__init__()
        self.input_size = input_size
        self.word_length = word_length
        self.hidden_size = hidden_size
        self.dictionary_size = dictionary_size
        self.device = device
        self.temperature = temperature

        if self.input_size is not None:
            self.encoder_embedding = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, input_size)

        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.latent_logits = nn.Linear(hidden_size, dictionary_size * word_length)

        self.decoder_embedding = nn.Linear(word_length * dictionary_size, hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward)
        self.decoder_transformer = nn.TransformerDecoder(self.decoder_layer, num_layers)

    def encode(self, x, sampling=True):
        src = self.encoder_embedding(x.to(self.device))
        src = src.unsqueeze(1)
        src = src.permute(1, 0, 2)
        self.encoded = self.encoder_transformer(src)
        logits = self.latent_logits(self.encoded).view(-1, self.word_length, self.dictionary_size)

        if sampling and self.training:
            z_sampled_soft = gumbel_softmax(logits, temperature=self.temperature)
        else:
            z_sampled_soft = torch.softmax(logits, dim=-1)

        one_hot_tokens, messages = straight_through_discretize(z_sampled_soft)
        return one_hot_tokens, logits, messages

    def decode(self, z):
        z = z.to(self.device)
        z = z.view(-1, self.word_length * self.dictionary_size)
        latent_embedded = F.relu(self.decoder_embedding(z))
        decoder_input = latent_embedded.unsqueeze(0)
        decoder_output = self.decoder_transformer(decoder_input, latent_embedded.unsqueeze(0))
        decoder_output = decoder_output.squeeze(0)
        recon = self.output(decoder_output)
        return recon

    def forward(self, input):
        one_hot_token, logit, message = self.encode(input)
        recon = self.decode(logit)
        return recon, one_hot_token, logit, message


class VAE_TFM2_img(VAE_TFM2):
    def __init__(self, word_length, dictionary_size, device, temperature=1.0,
                 hidden_size=200, num_heads=5, num_layers=2, dim_feedforward=2048):
        super(VAE_TFM2_img, self).__init__(
            input_size=None,
            word_length=word_length,
            dictionary_size=dictionary_size,
            device=device,
            temperature=temperature,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(64 * 8 * 8, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.latent_logits = nn.Linear(hidden_size, dictionary_size * word_length)

        self.decoder_embedding = nn.Linear(word_length * dictionary_size, hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward)
        self.decoder_transformer = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.decoder_fc = nn.Linear(hidden_size, 64 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # Output: (3, 64, 64)
            nn.Sigmoid()  # Assuming image pixel values are normalized to [0, 1]
        )

    def encode(self, x, sampling=True):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)
        x = self.encoder_transformer(x)
        logits = self.latent_logits(x).view(-1, self.word_length, self.dictionary_size)

        if sampling and self.training:
            z_sampled_soft = gumbel_softmax(logits, temperature=self.temperature)
        else:
            z_sampled_soft = torch.softmax(logits, dim=-1)

        one_hot_tokens, messages = straight_through_discretize(z_sampled_soft)
        return one_hot_tokens, logits, messages

    def decode(self, z):
        z = z.to(self.device)
        z = z.view(-1, self.word_length * self.dictionary_size)

        latent_embedded = F.relu(self.decoder_embedding(z))
        decoder_input = latent_embedded.unsqueeze(0)
        decoder_output = self.decoder_transformer(decoder_input, latent_embedded.unsqueeze(0))
        decoder_output = self.decoder_fc(decoder_output.squeeze(0))
        decoder_output = decoder_output.view(decoder_output.size(0), 64, 8, 8)  # Reshape to (batch_size, 64, 8, 8)
        recon = self.decoder_conv(decoder_output)
        return recon
