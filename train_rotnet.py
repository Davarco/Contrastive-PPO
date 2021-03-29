import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import argparse
import os
from PIL import Image


device = 'cuda'
writer = SummaryWriter(log_dir='logs')


class ProcgenDataset(Dataset):
    def __init__(self, data_dir):
        self.root = data_dir
        self.paths = os.listdir(data_dir)

    def __len__(self):
        return 4*len(self.paths)

    def __getitem__(self, index):
        obs = cv2.imread('{}/{}'.format(self.root, self.paths[index//4]))
        if index % 4 == 0:
            return obs.transpose(2, 0, 1)/255, 0
        if index % 4 == 1:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_90_CLOCKWISE)
            return obs.transpose(2, 0, 1)/255, 1
        if index % 4 == 2:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_180) 
            return obs.transpose(2, 0, 1)/255, 2
        if index % 4 == 3:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            return obs.transpose(2, 0, 1)/255, 3


class CIFARDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 4*len(self.dataset)

    def __getitem__(self, index):
        obs = np.array(self.dataset[index//4][0])
        if index % 4 == 0:
            return obs.transpose(2, 0, 1)/255, 0
        if index % 4 == 1:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_90_CLOCKWISE)
            return obs.transpose(2, 0, 1)/255, 1
        if index % 4 == 2:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_180) 
            return obs.transpose(2, 0, 1)/255, 2
        if index % 4 == 3:
            obs = cv2.rotate(obs, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            return obs.transpose(2, 0, 1)/255, 3


def create_model():
    model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            # 16384
            nn.Linear(4096, 1024),
            # nn.ReLU(),
            # nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4))
    return model


def train(model, optimizer, train_dl, validation_dl, epochs=1):
    train_accs = []
    val_accs = []

    it = 0
    for e in range(epochs):
        print('Epoch {}'.format(e))
        for t, (x, y) in enumerate(train_dl):
            model.train()
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 5 == 0:
                acc = evaluate(validation_dl, model)
                print('Iteration {}, loss = {}, acc = {}'.format(t, loss.item(), acc))
                writer.add_scalar('Training Loss', loss.item(), it)
                writer.add_scalar('Validation Accuracy', acc, it)
            it += 1
        val_accs.append(evaluate(validation_dl, model))
    return val_accs


def evaluate(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.shape[0]
        acc = float(num_correct) / num_samples
        # print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    args = vars(args)

    env_name = args['env_name']
    batch_size = args['batch_size']

    dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
    dataset = CIFARDataset(dataset)

    # dataset = ProcgenDataset('data/{}'.format(env_name))
    # print(dataset[0])
    # print(dataset[0][0].shape)
    # Image.fromarray((dataset[2][0]*255).astype(np.uint8).transpose(1, 2, 0)).show()
    train_size, validation_size, test_size = int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)
    train_data, validation_data, test_data = torch.utils.data.random_split(dataset, (train_size, validation_size, test_size))
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)

    model = create_model()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    validation_accs = train(model, optimizer, train_dl, validation_dl, epochs=10)
    
    test_acc = evaluate(test_dl, model)
    print(test_acc)

if __name__ == '__main__':
    main()


