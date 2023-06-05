import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, ToPILImage


class Maps(Dataset):

    def __init__(self, root, split, joint_transform=None):
        super(Maps, self).__init__()
        self.root = root
        self.split = split
        self.images_base = os.path.join(self.root, self.split)
        # self.masks_base = os.path.join(self.root, "masks", self.split)
        self.items_list = self._get_images_list()

        self.joint_transform = joint_transform

        mean_std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.image_transforms = Compose([ToTensor(), ColorJitter(), Normalize(*mean_std)])
        self.mask_transforms = Compose([ToTensor(), Normalize(*mean_std)])

    def _get_images_list(self):
        items_list = []
        for root, dirs, files in os.walk(self.images_base, topdown=True):

            for name in files:
                if name.endswith(".jpg"):
                    img_file = os.path.join(root, name)
                    items_list.append(img_file)
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]
        jpg = np.array(Image.open(image_path))
        image = jpg[:, :600, :]
        mask = jpg[:, 600:, :]


        if self.joint_transform:
            image, mask = self.joint_transform(Image.fromarray(image), Image.fromarray(mask))
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)


        return image, mask

    def __len__(self):
        return len(self.items_list)


def main():
    dataset = Maps("./dataset/maps", "val")
    print(len(dataset))
    dataiter = iter(dataset)
    image, mask = next(dataiter)
    print(image.shape, mask.shape)

    # transform = ToPILImage()
    # mask = transform(mask)
    # mask.show()

if __name__ == '__main__':
    main()