from __future__ import print_function
import random
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms


class LiveDataset:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_set = {}

    def add_image(self, image, label):
        train_image = self.convert_to_torch_image(image, self.device)
        if label in self.train_set:
            self.train_set[label] += [train_image]
        else:
            self.train_set[label] = [train_image]

    def get_batch(self, batch_size=16):
        labels = []
        images = []
        for i in range(batch_size):
            labels += [np.random.choice(list(self.train_set.keys()))]
            images += random.sample(self.train_set[labels[-1]], 1)
        return torch.stack(images), torch.Tensor(np.stack(labels)).long()

    @classmethod
    def convert_to_torch_image(self, image, device):
        train_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
        train_image = torch.Tensor(train_image.transpose(2, 0, 1) / 255).to(device)
        return train_image


def run_live():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    transforms = weights.transforms()
    for param in net.parameters():
        param.requires_grad = False
    for param in net.fc.parameters():
        param.requires_grad = True

    net.eval()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # TODO: Input the parameters you want to optimize over
    optimizer = optim.SGD(, lr=0.001)


    dataset = LiveDataset()
    train = False
    cnt = 0

    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, image = cap.read()

        output = net(LiveDataset.convert_to_torch_image(image, device).unsqueeze(0))
        output = output.argmax(1).cpu().numpy()[0]
        image = cv2.putText(image, 'label: ' + str(chr(output)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 50, 50), 3)
        cv2.imshow("img", image)
        key = cv2.waitKey(1)

        # TODO: Add images to your dataset
        # TODO: Run training on your dataset, while displaying the result





if __name__ == '__main__':
    run_live()
