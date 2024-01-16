import numpy as np
import torch
from torch.utils.data import DataLoader
import VAE_shapes3d
import VAE_dsprites
import gruVAE as languageVAE
from utils import DsrpitesDataset, Shapes3DDataset
const = 1e-7
beta = 1.0
scale = 0.1
path = '../data/'


class Agent:
    def __init__(self, name, args):
        super(Agent, self).__init__()
        self.name = name  # Agent name
        self.args = args
        self.runPath = args.run_path
        self.dataloader_object_train = None
        self.dataloader_object_test = None
        self.dataloader_latent = None
        self.true_label = []

        self.device = args.device
        self.D = args.D  # number of data points
        self.word_length = args.word_length
        self.latent_dim = args.latent_dim
        self.dictionary_size = args.dictionary_size
        self.batch_size = args.batch_size

        self.vae_perception = None
        self.vae_perception_train = None
        self.vae_perception_get_latent = None

        self.vae_language = None
        self.vae_language_train = None
        self.vae_language_get_message = None

        self.latents = []
        self.messages = []
        self.latents_pos = []
        self.acceptedCount = []
        self.mh_ratio_count = []
        self.initialize()

    def initialize(self):
        if self.args.dataset == 'dsprites':
            if self.name == 'a':
                data_link1 = path + 'dsprites/images_00.npy'
                data_link2 = path + 'dsprites/images_01.npy'
            else:
                data_link1 = path + 'dsprites/images_20.npy'
                data_link2 = path + 'dsprites/images_21.npy'

            label_link = '../data/dsprites/labels.npy'
            self.true_label = np.load(label_link)
            self.true_label = np.delete(self.true_label, [0, 3], axis=1)
            test_link = path + 'dsprites/images_10.npy'
            self.vae_perception = VAE_dsprites.VAE(latent_dim=self.latent_dim)
            self.vae_perception_train = VAE_dsprites.train
            self.vae_perception_get_latent = VAE_dsprites.get_latents
            dataset_train = DsrpitesDataset([data_link1, data_link2])
            dataset_test = DsrpitesDataset([test_link])

        else:       # 3dShapes
            if self.name == 'a':
                data_link1 = path + 'shapes3d/images_00.npy'
                data_link2 = path + 'shapes3d/images_01.npy'
                data_link3 = path + 'shapes3d/images_02.npy'
            else:
                data_link1 = path + 'shapes3d/images_14.npy'
                data_link2 = path + 'shapes3d/images_13.npy'
                data_link3 = path + 'shapes3d/images_12.npy'

            label_link = '../data/shapes3d/labels.npy'
            self.true_label = np.load(label_link)
            self.true_label = self.true_label[:, :-1]
            test_link = path + 'shapes3d/images_07.npy'
            self.vae_perception = VAE_shapes3d.VAE(latent_dim=self.latent_dim)
            self.vae_perception_train = VAE_shapes3d.train
            self.vae_perception_get_latent = VAE_shapes3d.get_latents
            dataset_train = Shapes3DDataset([data_link1, data_link2, data_link3])
            dataset_test = Shapes3DDataset([test_link])

        self.dataloader_object_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False)
        self.dataloader_object_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        self.vae_language = languageVAE.VAE(input_size=self.latent_dim, word_length=self.word_length,
                                            dictionary_size=self.dictionary_size, device=self.device, temperature=1.0)
        self.vae_language_train = languageVAE.train
        self.vae_language_get_message = languageVAE.get_messages

    def train_vae_object(self):
        self.vae_perception.to(self.device)
        self.vae_perception_train(model=self.vae_perception, dataloader=self.dataloader_object_train,
                                  learning_rate=self.args.learning_rate, device=self.device,
                                  epochs=self.args.vae1_epochs, beta=self.args.vae_perception_beta,
                                  saved=self.runPath+self.name+'_vae_perception.pth')
        self.latents = self.vae_perception_get_latent(model=self.vae_perception,
                                                      dataloader=self.dataloader_object_train, device=self.device)
        self.dataloader_latent = DataLoader(self.latents, batch_size=self.batch_size, shuffle=False)

    def train_vae_language(self):
        self.vae_language.to(self.device)
        self.vae_language_train(model=self.vae_language, dataloader=self.dataloader_latent,
                                learning_rate=self.args.learning_rate, device=self.device,
                                epochs=self.args.vae2_epochs, saved=self.runPath+self.name+'_vae_language.pth')
        self.messages = self.vae_language_get_message(model=self.vae_language, dataloader=self.dataloader_latent,
                                                      device=self.device)

    def train_MH_languageVAE(self, Speaker, optimizer, mode=1):
        self.vae_language.train()
        Speaker.vae_language.eval()
        train_loss, train_reco, train_KLD = 0, 0, 0
        accept_count = 0
        interval_counts = [0] * 12
        for data, Sp_data in zip(self.dataloader_latent, Speaker.dataloader_latent):
            data = data.float().to(self.device)
            Sp_data = Sp_data.float().to(Speaker.device)
            optimizer.zero_grad()

            messages, _, _ = self.vae_language.Encoder(data)
            Sp_messages, _, _ = Speaker.vae_language.Encoder(Sp_data)
            recon, recon_mean, recon_logvar = self.vae_language.Decoder(messages)
            Sp_recon, Sp_recon_mean, Sp_recon_logvar = self.vae_language.Decoder(Sp_messages)

            if mode == 3:           # All accepted
                loss, recon_loss, kld_loss = self.vae_language.elbo(data, Sp_recon, Sp_messages)
                accept_count = self.D

            elif mode == 2:        # All rejected
                loss, recon_loss, kld_loss = self.vae_language.elbo(data, recon, messages)

            else:                   # MH Naming game
                MH_rate = compute_2Gaussian_ratio(recon, recon_mean, recon_logvar,
                                                  Sp_recon, Sp_recon_mean, Sp_recon_logvar)
                for num in MH_rate:
                    if num == 0:
                        interval_counts[0] += 1
                    elif num == 1:
                        interval_counts[11] += 1
                    else:
                        interval_index = int(num * 10) + 1
                        interval_counts[interval_index] += 1

                judge_r = torch.minimum(torch.tensor(1.0).to(MH_rate.device), MH_rate)
                rand_u = torch.tensor(np.random.rand(MH_rate.size(0))).to(MH_rate.device)

                # Create masks based on the acceptance criterion
                accept_mask = judge_r >= rand_u
                accepted = torch.sum(accept_mask).item()
                accept_count += accepted

                if accept_mask.all():  # All are accepted
                    loss_accept, recon_loss_accept, kld_loss_accept = self.vae_language.elbo(data, Sp_recon, Sp_messages)
                    loss_reject, recon_loss_reject, kld_loss_reject = 0, 0, 0
                elif (~accept_mask).all():  # All samples are rejected
                    loss_reject, recon_loss_reject, kld_loss_reject = self.vae_language.elbo(data, recon, messages)
                    loss_accept, recon_loss_accept, kld_loss_accept = 0, 0, 0
                else:
                    loss_accept, recon_loss_accept, kld_loss_accept = self.vae_language.elbo(data[accept_mask], Sp_recon[accept_mask], Sp_messages[accept_mask])
                    loss_reject, recon_loss_reject, kld_loss_reject = self.vae_language.elbo(data[~accept_mask], recon[~accept_mask], messages[~accept_mask])
                loss = loss_accept + loss_reject
                recon_loss = recon_loss_accept + recon_loss_reject
                kld_loss = kld_loss_accept + kld_loss_reject

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_reco += recon_loss.item()
            train_KLD += kld_loss.item()

        self.mh_ratio_count.append(interval_counts)
        self.acceptedCount.append(accept_count)
        print(self.name + ' - Reco: {:.8f}, KLD: {:.8f}, Accept: {}'.format(train_reco / self.D, train_KLD / self.D, accept_count))


def multivariate_gaussian_logpdf(x, mu, logvar):
    pi = torch.tensor(3.14159265358979323846)
    log_coefficient = -0.5 * (logvar + torch.log(2.0 * pi))
    exponential_term = - (x - mu) ** 2 / (2 * torch.exp(logvar))
    joint_logprob = torch.sum(log_coefficient + exponential_term, dim=1)
    return joint_logprob


def compute_2Gaussian_ratio(recon_Li, mean_Li, logvar_Li, recon_Sp, mean_Sp, logvar_Sp):
    term_recon_Li = multivariate_gaussian_logpdf(recon_Li, mean_Li, logvar_Li)
    term_recon_Sp = multivariate_gaussian_logpdf(recon_Sp, mean_Sp, logvar_Sp)
    log_R = term_recon_Sp - term_recon_Li
    R = torch.exp(log_R)
    R = torch.where(R > 1, torch.tensor(1.0, device=R.device, dtype=R.dtype), R)
    return R

