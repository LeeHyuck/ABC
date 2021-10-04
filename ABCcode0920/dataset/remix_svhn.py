import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault
from torchvision.transforms import ToPILImage
SVHN_mean = (0.4377, 0.4438, 0.4728)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
SVHN_std = (0.1980, 0.2010, 0.1970)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(SVHN_mean, SVHN_std)


])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(SVHN_mean, SVHN_std)

])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
transforms.Normalize(SVHN_mean, SVHN_std)
])
def normalise(x, mean=SVHN_mean, std=SVHN_std,source='NCHW', target='NHWC'):
    x=x.transpose([source.index(d) for d in target])
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3


def get_SVHN(root, l_samples, u_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):
    base_dataset = torchvision.datasets.SVHN(root, split='train', download=download)
    base_dataset2 = torchvision.datasets.SVHN(root, split='test', download=download)
    test_idxs = testsplit(base_dataset2.labels)
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.labels, l_samples, u_samples)

    train_labeled_dataset = SVHN_labeled(root, train_labeled_idxs, split='train', transform=transform_strong)
    train_unlabeled_dataset = SVHN_unlabeled(root, train_unlabeled_idxs, split='train',
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = SVHN_labeled(root,test_idxs, split='test', transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset,test_dataset
def testsplit(labels):
    labels = np.array(labels)
    test_idxs=[]
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        test_idxs.extend(idxs[:1500])
    np.random.shuffle(test_idxs)
    return test_idxs

def train_split(labels, n_labeled_per_class, n_unlabeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(10):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs


class SVHN_labeled(torchvision.datasets.SVHN):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(SVHN_labeled, self).__init__(root, split=split,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        #print(np.shape(self.data))
        self.data = transpose(self.data)
        # print(np.shape(self.data))

        self.data = [Image.fromarray(img) for img in self.data]

        #print(np.shape(self.data))
        #
        #self.data = [ToPILImage()((img * 255).astype(np.uint8)) for img in self.data]
        #print(len(self.data),np.shape(self.data[0]))


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(SVHN_unlabeled, self).__init__(root, indexs, split=split,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.labels = np.array([-1 for i in range(len(self.labels))])