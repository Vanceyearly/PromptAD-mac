import os

import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from utils.perlin import perlin_noise, get_forehead_mask, perlin_noise_2
from datasets.cutpaste import CutPaste

class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot):

        self.load_function = load_function
        self.phase = phase

        self.category = category

        self.cutpaste_transform = CutPaste(transform=True, type='binary')

        # load datasets
        # self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot)  # self.labels => good : 0, anomaly : 1
        self.img_paths, self.gt_paths, self.labels, self.types, self.dtd_paths = self.load_dataset(k_shot)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, k_shot):

        # (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        # (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
        #                                                                                               k_shot)
        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types), \
            dtd_img_paths = self.load_function(self.category, k_shot)
        
        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types, \
                   dtd_img_paths,
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types, dtd_img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        dtd_path = self.dtd_paths[idx] if hasattr(self, 'dtd_paths') else None
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[0]])
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255

        img = cv2.resize(img, (1024, 1024))
        gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        # perlin_noise
        noise_img = cv2.imread(dtd_path, cv2.IMREAD_COLOR) if dtd_path else None
        if noise_img is not None:
            noise_img = cv2.resize(noise_img, (1024, 1024))
            # perlin_image, aug_mask = perlin_noise(img, noise_img, aug_prob=1.0, mask=None)

            ori_image, cutpaste_image = self.cutpaste_transform(img) if random.random() < 0.5 else (img, img)
            threshold_slightly = random.uniform(0.8, 0.9)
            threshold_median = random.uniform(0.5, 0.8)
            threshold_significantly = random.uniform(0.1, 0.5)
            beta_slightly = random.uniform(0.7, 0.9)
            beta_median = random.uniform(0.4, 0.7)
            beta_significantly = random.uniform(0.1, 0.4)

            # # slightly
            perlin_slightly, aug_mask_slightly = perlin_noise_2(cutpaste_image, noise_img, aug_prob=1.0, mask=None, perlin_threshold=threshold_slightly, blend_beta=beta_slightly, random_beta=True)
            # # median
            perlin_median, aug_mask_median = perlin_noise_2(cutpaste_image, noise_img, aug_prob=1.0, mask=None, perlin_threshold=threshold_median, blend_beta=beta_median, random_beta=True)
            # # significantly
            perlin_significantly, aug_mask_significantly = perlin_noise_2(cutpaste_image, noise_img, aug_prob=1.0, mask=None, perlin_threshold=threshold_significantly, blend_beta=beta_significantly, random_beta=True)

            # perlin_image = (perlin_image * 255).clip(0, 255).astype(np.uint8)
            perlin_image = [
                (perlin_slightly * 255).clip(0, 255).astype(np.uint8),
                (perlin_median * 255).clip(0, 255).astype(np.uint8),
                (perlin_significantly * 255).clip(0, 255).astype(np.uint8)
            ]
            aug_mask = [
                aug_mask_slightly,
                aug_mask_median,
                aug_mask_significantly
            ]
        
        # cutpaste
        # ori_image, cutpaste_image = self.cutpaste_transform(img)

        return img, gt, label, img_name, img_type, perlin_image, aug_mask
