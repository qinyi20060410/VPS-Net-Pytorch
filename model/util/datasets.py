import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

import glob
import random
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from util.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from util.utils import *


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=600):
        self.files = sorted(glob.glob("%s/*.jpg" % folder_path))
        self.mat_files = sorted(glob.glob("%s/*.mat" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        mat_path = self.mat_files[index % len(self.mat_files)]
        matfile = sio.loadmat(mat_path)
        slot = matfile['slots']
        mark = matfile['marks']

        return img_path, img, slot, mark

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self,
                 list_path,
                 img_size=416,
                 augment=True,
                 multiscale=True,
                 normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images",
                         "labels").replace(".png",
                                           ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class PS_Dataset(Dataset):
    def __init__(self,
                 folder_path,
                 img_size=416,
                 augment=True,
                 multiscale=False,
                 normalized_labels=True):
        super(PS_Dataset, self).__init__()
        self.sample_names = []
        self.root = folder_path
        self.img_size = img_size
        self.augment = augment
        self.normalized_labels = normalized_labels
        for file in os.listdir(self.root):
            if file.endswith(".mat"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        img = transforms.ToTensor()(Image.open(
            os.path.join(self.root, name + '.jpg')).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # generate points x1,y1,x2,y2
        matfile = sio.loadmat(os.path.join(self.root, name + '.mat'))
        slot = matfile['slots']
        mark = matfile['marks']
        mark_point = []
        slot_point = []

        for data in mark:
            x_ = data[0] / padded_w
            y_ = data[1] / padded_h
            w_ = 60 / padded_w
            h_ = 60 / padded_h
            mark_point.append(np.array([3, x_, y_, w_, h_]))

        for data in slot:
            x = (mark[int(data[0] - 1)][0] +
                 mark[int(data[1] - 1)][0]) / 2.0 / padded_w
            y = (mark[int(data[0] - 1)][1] +
                 mark[int(data[1] - 1)][1]) / 2.0 / padded_h
            w = (max(mark[int(data[0] - 1)][1], mark[int(data[1] - 1)][1]) -
                 min(mark[int(data[0] - 1)][1], mark[int(data[1] - 1)][1]) +
                 60) / padded_w
            h = (max(mark[int(data[0] - 1)][0], mark[int(data[1] - 1)][0]) -
                 min(mark[int(data[0] - 1)][0], mark[int(data[1] - 1)][0]) +
                 60) / padded_h

            angle_type = None
            if data[3] == 90:
                angle_type = 0
            elif data[3] < 90:
                angle_type = 1
            elif data[3] > 90:
                angle_type = 2

            slot_point.append(np.array([angle_type, x, y, w, h]))

        point_label = torch.cat(
            (torch.from_numpy(np.array(mark_point, dtype=np.float32)),
             torch.from_numpy(np.array(slot_point, dtype=np.float32))),
            dim=0)

        targets = torch.zeros((point_label.shape[0], 6))
        targets[:, 1:] = point_label

        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img, targets

    def __len__(self):
        return len(self.sample_names)

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        targets = torch.cat(targets, 0)
        return imgs, targets


class PS_Dataset_C(Dataset):
    def __init__(self,
                 folder_path,
                 img_size=416,
                 augment=True,
                 multiscale=False,
                 normalized_labels=True):
        super(PS_Dataset_C, self).__init__()
        self.sample_names = []
        self.root = folder_path
        self.img_size = img_size
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.normalized_labels = normalized_labels
        for file in os.listdir(self.root):
            if file.endswith(".mat"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        # print(name)
        img = cv2.imread(os.path.join(self.root, name + '.jpg'))

        # generate points x1,y1,x2,y2
        matfile = sio.loadmat(os.path.join(self.root, name + '.mat'))
        slot = matfile['slots']
        mark = matfile['marks']
        imgs = []
        labels = []

        for data in slot:
            x1 = mark[int(data[0] - 1)][0]
            y1 = mark[int(data[0] - 1)][1]
            x2 = mark[int(data[1] - 1)][0]
            y2 = mark[int(data[1] - 1)][1]
            angle = data[3]

            if data[2] <= 2:
                label = torch.from_numpy(np.array([data[2]
                                                   ])).type(torch.LongTensor)

                point1 = np.array([x1, y1])
                point2 = np.array([x2, y2])

                pts = compute_four_points(angle, point1, point2)
                point3_org = copy.copy(pts[2])
                point4_org = copy.copy(pts[3])

                regul_img = image_preprocess(img, pts)
                regul_img = self.transform(regul_img)

                imgs.append(regul_img)
                labels.append(label)

        return imgs, labels

    def __len__(self):
        return len(self.sample_names)

    def collate_fn(self, batch):
        imgs_set, labels_set = list(zip(*batch))
        images = None
        targets = None
        if len(imgs_set) > 0:
            for imgs in imgs_set:
                if len(imgs) > 0:
                    temp = [img for img in imgs]
                    images = torch.stack(temp)
        if len(labels_set) > 0:
            # print('set', labels_set)
            for labels in labels_set:
                if len(labels) > 0:
                    targets = [label for label in labels]
                    targets = torch.cat(targets, 0)
                    # targets = torch.stack(temp)

        return images, targets


if __name__ == '__main__':
    root = 'data/training'
    sample_names = []
    for file in os.listdir(root):
        if file.endswith(".mat"):
            sample_names.append(os.path.splitext(file)[0])

    for index in range(len(sample_names)):
        name = sample_names[index]
        print('-------------------------------------------------------')
        print(os.path.join(root, name + '.jpg'))
        img = cv2.imread(os.path.join(root, name + '.jpg'))

        # generate points x1,y1,x2,y2
        matfile = sio.loadmat(os.path.join(root, name + '.mat'))
        slot = matfile['slots']
        mark = matfile['marks']
        mark_point = []
        slot_point = []

        for i in range(slot.shape[0]):
            data = slot[i]

            print('mark', mark)
            print('slot', slot)
            print('angle', data[3])

            if data[2] >= 3:

                x1 = mark[int(data[0] - 1)][0]
                y1 = mark[int(data[0] - 1)][1]
                x2 = mark[int(data[1] - 1)][0]
                y2 = mark[int(data[1] - 1)][1]
                angle = data[3]

                point1 = np.array([x1, y1])
                point2 = np.array([x2, y2])

                cv2.circle(img, (round(float(x1)), round(float(y1))), 6,
                           (255, 0, 255), 2, 8)
                cv2.circle(img, (round(float(x2)), round(float(y2))), 6,
                           (255, 0, 255), 2, 8)

                pts = compute_four_points(angle, point1, point2)
                point3_org = copy.copy(pts[2])
                point4_org = copy.copy(pts[3])

                regul_img = image_preprocess(img, pts)

                pts_show = np.array([pts[0], pts[1], point3_org, point4_org],
                                    np.int32)

                cv2.polylines(img, [pts_show], True, (255, 0, 0), 2)

                cv2.imshow('crop', regul_img)
                cv2.imshow("Image", img)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
                # elif cv2.waitKey(0) & 0xFF == ord('n'):
                #     continue
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            continue

    cv2.destroyAllWindows()

    # train_dataset = PS_Dataset_C('data/training')
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=True,
    #     collate_fn=train_dataset.collate_fn)

    # for i, (imgs, targets) in enumerate(train_dataloader):
    #     print('index', i)
    #     print('imgs', imgs)
    #     print('targets', targets)
    # print(len(imgs))
    # print(imgs)
    # print(labels)
