import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import random
import h5py
link_dsprites = '../data/dsprites/'
link_shapes3d = '../data/shapes3d/'


class DSpritesDataset(Dataset):
    def __init__(self, transform=None):
        file_path = '../data/dsprites/dsprites_dataset.npz'
        data = np.load(file_path)
        self.imgs = data['imgs'].astype(np.float32)
        self.latents_values = data['latents_values']
        self.latents_classes = data['latents_classes']
        self.transform = transform
        self.set_indices = self.find_angle_sets()
        self.size = 737280

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx][None, :, :]
        image = Image.fromarray((image.squeeze() * 255).astype(np.uint8), mode='L')
        if self.transform:
            image = self.transform(image)

        latent_values = self.latents_values[idx]
        latent_classes = self.latents_classes[idx]
        return image, latent_values, latent_classes

    def find_angle_sets(self):
        angles = range(40)  # Angles from 0 to 39
        angle_indices = {angle: [] for angle in angles}
        for i, latent_class in enumerate(self.latents_classes):
            angle = latent_class[3]  # Angle is in the 3th column
            angle_indices[angle].append(i)
        return angle_indices

    def save_sets(self):
        for angle in self.find_angle_sets():
            imgs = [self.imgs[idx] for idx in self.find_angle_sets()[angle]]
            file_name = f'images_{angle:02}.npy'
            np.save(file_name, np.array(imgs))


def display_image_sets_dsprites(num_pairs=10, *img_files):
    all_images = [np.load(file) for file in img_files]
    min_size = min(images.shape[0] for images in all_images)
    indices = random.sample(range(min_size), num_pairs)
    fig, axes = plt.subplots(len(img_files), num_pairs, figsize=(40, 4 * len(img_files)))

    if len(img_files) == 1:
        axes = [axes]
    for row, images in enumerate(all_images):
        for col, idx in enumerate(indices):
            axes[row][col].imshow(images[idx], cmap='gray')
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()

# dataset = DSpritesDataset(transform=None)
# dataset.save_sets()


class Shapes3DDataset(Dataset):
    def __init__(self, transform=None):
        file_path = '../data/shapes3d/3dshapes.h5'
        with h5py.File(file_path, 'r') as dataset:
            self.imgs = dataset['images'][:]
            self.labels = dataset['labels'][:]
        self.transform = transform
        self.size = 480000
        self.set_indices = self.find_angle_sets()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx].astype(np.float32)  # Convert images to float32
        image = Image.fromarray(image.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        image = np.transpose(image, (2, 0, 1))
        labels = self.labels[idx]
        return image, labels

    def find_angle_sets(self):
        angles = [-30.0, -25.714285714285715, -21.42857142857143, -17.142857142857142, -12.857142857142858,
                  -8.571428571428573, -4.285714285714285, 0.0, 4.285714285714285, 8.57142857142857,
                  12.857142857142854, 17.14285714285714, 21.42857142857143, 25.714285714285715, 30.0]
        angle_indices = {angle: [] for angle in angles}
        for i, label in enumerate(self.labels):
            angle = label[5]
            if angle in angles:
                angle_indices[angle].append(i)
        return angle_indices

    def save_sets(self):
        for angle in self.find_angle_sets():
            imgs = [self.imgs[idx] for idx in self.find_angle_sets()[angle]]
            file_name = f'images_{int((angle + 30) / 4.285714285714285):02}.npy'
            np.save(file_name, np.array(imgs))


def display_image_sets_shape3d(num_pairs=10, *img_files):
    all_images = [np.load(file) for file in img_files]
    min_size = min(images.shape[0] for images in all_images)
    indices = random.sample(range(min_size), num_pairs)
    fig, axes = plt.subplots(len(img_files), num_pairs, figsize=(40, 4 * len(img_files)))

    if len(img_files) == 1:
        axes = [axes]
    for row, images in enumerate(all_images):
        for col, idx in enumerate(indices):
            axes[row][col].imshow(images[idx])
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()

# dataset = Shapes3DDataset(transform=None)
# dataset.save_sets()
