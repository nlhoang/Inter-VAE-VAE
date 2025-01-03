import random
import sys
import os
import shutil
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def set_seeds(seed):
    if seed == -1:
        seed = random.randint(1, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed: {:.2g}'.format(seed))


class DsrpitesDataset(Dataset):
    def __init__(self, npy_files):
        self.data = np.concatenate([np.load(file) for file in npy_files], axis=0)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class Shapes3DDataset(Dataset):
    def __init__(self, npy_files, resnet=False):
        self.data = np.concatenate([np.load(file) for file in npy_files], axis=0)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224) if resnet else (64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        sample = Image.fromarray(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    return checkpoint


def param_count(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def visualize_ls(means, labels, save_dir, description):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    points_pca = PCA(n_components=2, random_state=0).fit_transform(means)
    points_tsne = TSNE(n_components=2, random_state=0).fit_transform(means)

    # TSNE
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_tsne, labels):
        plt.title("TSNE", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'tsne_' + description + '.png')
    plt.close()

    # PCA
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_pca, labels):
        plt.title("PCA", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'pca_' + description + '.png')
    plt.close()


def visualize_tsne(means, labels, save_dir, description):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    points_tsne = TSNE(n_components=2, random_state=0).fit_transform(means)

    # TSNE
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_tsne, labels):
        plt.title("TSNE", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'tsne_' + description + '.png')
    plt.close()


def visualize_pca(means, labels, save_dir, description):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    points_pca = PCA(n_components=2, random_state=0).fit_transform(means)

    # PCA
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_pca, labels):
        plt.title("PCA", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'pca_' + description + '.png')
    plt.close()


def save_toFile(path, file_name, data_saved, rows=0):
    f = open(path + file_name, 'w')
    writer = csv.writer(f)
    if rows == 0:
        writer.writerow(data_saved)
    if rows == 1:
        writer.writerows(data_saved)
    f.close()


def figure(data_list1, data_list2, label1, label2, data_size, save_path=None):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    a = [(item / data_size) * 100 for item in data_list1]
    ax.plot(a, label=label1)
    b = [(item / data_size) * 100 for item in data_list2]
    ax.plot(b, label=label2)

    plt.xlabel('Epochs')
    plt.ylabel('Acceptance rate (%)')
    plt.title('Success Rate of MH naming game')
    plt.ylim(0, 100)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def mh_count_heatmap(data, save_path=None):
    selected_data = [data[i] for i in range(0, len(data), 20)]
    selected_data.append(data[-1])
    item_labels = [str(i) for i in range(0, len(data), 20)]
    item_labels.append('500')
    sublist_labels = ['=0'] + [f'({i / 10},{(i + 1) / 10}]' for i in range(0, 10)] + ['=1']

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(selected_data, annot=True, fmt=".0f", cmap="viridis")
    ax.set_yticklabels(item_labels, rotation=0)
    ax.set_xticklabels(sublist_labels, rotation=45)
    plt.title("Heatmap of MH ratio values")
    plt.xlabel("MH ratio value ranges")
    plt.ylabel("Epochs (every 20th)")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

