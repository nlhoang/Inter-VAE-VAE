import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_model import gumbel_softmax, straight_through_discretize


def forward_with_attention(encoder, src, num_heads, hidden_size):
    """
    Forward pass through Transformer Encoder, capturing detailed attention weights for each head.

    Args:
        encoder (nn.TransformerEncoder): Transformer encoder instance.
        src (torch.Tensor): Input tensor to the encoder.
        num_heads (int): Number of attention heads.
        hidden_size (int): Size of the hidden dimension.

    Returns:
        torch.Tensor: Encoded representation.
        list of torch.Tensor: List of attention weights from each layer, shape: (batch_size, num_heads, seq_len, seq_len).
    """
    attn_weights = []

    for layer in encoder.layers:
        q_proj_weight = layer.self_attn.in_proj_weight[:hidden_size]
        k_proj_weight = layer.self_attn.in_proj_weight[hidden_size:2*hidden_size]
        v_proj_weight = layer.self_attn.in_proj_weight[2*hidden_size:]

        q_proj_bias = layer.self_attn.in_proj_bias[:hidden_size]
        k_proj_bias = layer.self_attn.in_proj_bias[hidden_size:2*hidden_size]
        v_proj_bias = layer.self_attn.in_proj_bias[2*hidden_size:]

        q = F.linear(src, q_proj_weight, q_proj_bias)
        k = F.linear(src, k_proj_weight, k_proj_bias)
        v = F.linear(src, v_proj_weight, v_proj_bias)

        seq_len, batch_size, _ = q.size()  # Reshape for multi-head attention
        head_dim = hidden_size // num_heads

        q = q.view(seq_len, batch_size, num_heads, head_dim).transpose(1, 2)  # (seq_len, num_heads, batch_size, head_dim)
        k = k.view(seq_len, batch_size, num_heads, head_dim).transpose(1, 2)
        v = v.view(seq_len, batch_size, num_heads, head_dim).transpose(1, 2)

        q = q.permute(2, 1, 0, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.permute(2, 1, 0, 3)
        v = v.permute(2, 1, 0, 3)

        scaling = float(head_dim) ** -0.5  # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights_layer = F.softmax(attn_scores, dim=-1)  # Softmax over the seq_len dimension
        attn_weights.append(attn_weights_layer.detach().cpu())
        attn_output = torch.matmul(attn_weights_layer, v)  # Compute attention output (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_len, batch_size, hidden_size)  # Concatenate attention heads and reshape back to (seq_len, batch_size, hidden_size)

        src = layer.norm1(src + layer.dropout1(attn_output))  # Continue with the rest of the transformer encoder logic
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
        src = layer.norm2(src + layer.dropout2(src2))

    return src, attn_weights


class VAE(nn.Module):
    def __init__(self, input_size, word_length, dictionary_size, device, temperature=1.0,
                 hidden_size=200, num_heads=5, num_layers=2, dim_feedforward=2048):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.word_length = word_length
        self.hidden_size = hidden_size
        self.dictionary_size = dictionary_size
        self.device = device
        self.temperature = temperature
        self.num_heads = num_heads
        self.sub_input = 10
        self.encoder_embedding = nn.Linear(self.sub_input, hidden_size)
        self.encoder = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder, num_layers)
        self.latent_logits = nn.Linear(hidden_size, dictionary_size * word_length)

        self.decoder_fc1 = nn.Linear(word_length * dictionary_size, hidden_size * 3)
        self.decoder_fc2 = nn.Linear(hidden_size * 3, hidden_size)
        self.output_mean = nn.Linear(hidden_size, input_size)
        self.output_logvar = nn.Linear(hidden_size, input_size)

    def Encoder(self, x, sampling=True):
        x = x.to(self.device)  # x.shape = (batch_size, input_size)
        seq_length = self.input_size // self.sub_input
        x = x.view(x.size(0), seq_length, self.sub_input)  # Shape = (batch_size, seq_length, token_size)
        src = self.encoder_embedding(x)  # Shape = (batch_size, seq_length, hidden_size)
        src = src.permute(1, 0, 2)  # Shape = (seq_length, batch_size, hidden_size)

        if self.training:
            encoded_output = self.encoder_transformer(src)
            attn_weights = None
        else:
            encoded_output, attn_weights = forward_with_attention(self.encoder_transformer, src, self.num_heads, self.hidden_size)

        # encoded_final_token = encoded_output[-1]  # Option 1: Use the last token - Shape: (batch_size, hidden_size)
        encoded_final_token = encoded_output.mean(dim=0)  # Option 2: Use mean pooling - Shape: (batch_size, hidden_size)
        # encoded_final_token = encoded_output.max(dim=0).values  # Option 3: Use max pooling - Shape: (batch_size, hidden_size)
        logits = self.latent_logits(encoded_final_token).view(-1, self.word_length, self.dictionary_size)

        if sampling and self.training:
            z_sampled_soft = gumbel_softmax(logits, temperature=self.temperature)
        else:
            z_sampled_soft = torch.softmax(logits, dim=-1)

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
        prior = torch.log(torch.tensor([1.0 / self.dictionary_size] * self.dictionary_size, device=self.device))
        prior_dist = torch.distributions.OneHotCategorical(logits=prior.expand_as(logits))
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        return kl.sum(1).mean()


def get_messages(model, dataloader, device):
    model.eval()
    latent, message = [], []
    # attn_weights_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.float().to(device)
            recon, _, logits, mes = model(data)
            latent.append(recon.cpu().numpy())
            message.append(mes.cpu().numpy())

            # batch_size = data.size(0)  # 256 in this case
            # for i in range(batch_size):  # Iterate over each sample in the batch
            #     attn_weights_sample = [attn_layer[i].cpu().numpy() for attn_layer in attn_weights]
            #     attn_weights_list.append(attn_weights_sample)

    latent = np.concatenate(latent, axis=0)
    message = np.concatenate(message, axis=0)
    return latent, message


def main():
    from torch.utils.data import DataLoader
    from utils import param_count, save_toFile
    from VAE_lang_gru import train, get_messages

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    batch_size = 64
    lr = 1e-3
    word_length = 10
    dictionary_size = 100
    input_size = 50
    data_link = 'pretrained/latents_dsprites_VAE_lin_d50_a.npy'
    model_saved = 'pretrained/model_dsprites_test.pth'
    data_saved = 'messages_dsprites_test.npy'
    dataset = np.genfromtxt(data_link, delimiter=',', dtype=float)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset, batch_size=18432, shuffle=False)
    model = VAE(input_size=input_size, word_length=word_length, dictionary_size=dictionary_size, device=device,
                temperature=1.0, hidden_size=100, num_heads=5, num_layers=2, dim_feedforward=2048).to(device)
    print('Model Size: {}'.format(param_count(model)))
    train(model=model, dataloader=dataloader_train, learning_rate=lr, device=device, epochs=epochs, saved=model_saved)
    _, messages = get_messages(model, dataloader_test, device)
    print(messages)
    save_toFile('experiments/', data_saved, messages, rows=1)
    # np.savez("attn_weights.npz", attn_weights_list=attn_weights_list)
    # data = np.load("attn_weights.npz", allow_pickle=True)
    # attn_weights_list_saved = data["attn_weights_list"]


if __name__ == "__main__":
    main()
