import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import argparse
import os


class ProcgenDataset(Dataset):
    def __init__(self, data_dir):
        self.root = data_dir
        self.paths = os.listdir(data_dir)

    def __len__(self):
        return 4*len(self.paths)

    def __getitem__(self, index):
        obs = cv2.imread('{}/{}'.format(self.root, self.paths[index//4]))
        if index % 4 == 0:
            return obs.transpose(2, 0, 1), 0
        if index % 4 == 1:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_90_CLOCKWISE)
            return obs.transpose(2, 0, 1), 1
        if index % 4 == 2:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_180) 
            return obs.transpose(2, 0, 1), 2
        if index % 4 == 3:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            return obs.transpose(2, 0, 1), 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name')
    args = parser.parse_args()
    args = vars(args)

    env_name = args['env_name']

    dataset = ProcgenDataset('data/{}'.format(env_name))


if __name__ == '__main__':
    main()
