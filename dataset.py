import os
import cv2
import pandas
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import functional as F


class ContainersDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train'):
        super(ContainersDataset).__init__()
        self.image_folder_path = os.path.join(dataset_path, 'img')
        image_names = os.listdir(self.image_folder_path)
        self.split = split
        self.train, self.test = train_test_split(image_names, test_size=0.1, random_state=42, shuffle=True)
        dataframe = pandas.read_excel(os.path.join(dataset_path, 'data.xlsx'))
        self.labels = dict(dataframe.values)
        del dataframe
        self.keys = list(set(self.labels.values()))
        self.keys.sort()

    def __getitem__(self, item):
        target_list = self.train if self.split == 'train' else self.test
        image_name = target_list[item]

        image_path = os.path.join(self.image_folder_path, image_name)
        img = cv2.imread(image_path)

        dim = (300, 450)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = F.to_tensor(resized)

        label = self.keys.index(self.labels[image_name])
        label = torch.as_tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        target_list = self.train if self.split == 'train' else self.test
        return len(target_list)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    dataset = ContainersDataset(dataset_path='/home/home/PycharmProjects/datasets/containers')
    DL_train = torch.utils.data.DataLoader(dataset=dataset, batch_size=4,
                                           shuffle=True, collate_fn=collate_fn)

    print('Success!')
