import time

import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, PILToTensor, Compose, ConvertImageDtype, Resize, Lambda, InterpolationMode
from torchvision import models

from torch.nn import Conv2d, Linear, Dropout, AvgPool2d, MaxPool2d
from torch.nn.functional import relu, elu

if torch.cuda.is_available():
    device = 'cuda'
    device_nr = torch.cuda.current_device()
    print(f'Found GPU device: {device_nr} of type: {torch.cuda.get_device_name(device)}')
else:
    device = 'cpu'
    print('No GPU found')


class FinetuneNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FinetuneNet, self).__init__()
        self.num_classes = num_classes
        # TODO: Initialize the layers of your network
        # You can find different layers in tensorflow.keras.layers (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers)
        weights = models.ResNet18_Weights.DEFAULT
        self.base_model = models.resnet18(weights=weights)
        self.transforms = weights.transforms()
        # self.transforms = lambda x: x
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        # for param in self.base_model.layer4.parameters():
        #     param.requires_grad = True
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = Linear(num_ftrs, num_classes)

    def forward(self, x, output_features=False):
        # TODO: Run the image through your network
        # Your input should be a [Batch_size x 3 x 32 x 32] sized tensor
        # Your output should be a [Batch_size x num_classes] sized matrix
        x = self.base_model(self.transforms(x))
        if output_features: return x
        # Return the result of your network
        return x


class SimpleNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        # TODO: Initialize the layers of your network
        # You can find different layers in tensorflow.keras.layers (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers)
        padding = 'same'
        self.pool = MaxPool2d((2, 2), stride=(2, 2))
        self.conv1 = torch.nn.Conv2d(3, 32, 5, padding=padding)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=padding)
        self.conv3 = Conv2d(64, 128, 5, padding=padding)
        self.conv4 = Conv2d(128, 128, 5, padding=padding)
        self.dropout = Dropout(0.5)
        self.conv5 = Conv2d(128, num_classes, 8, stride=1, padding='valid')

    def forward(self, x, output_features=False):
        # TODO: Run the image through your network
        # Your input should be a [Batch_size x 3 x 32 x 32] sized tensor
        # Your output should be a [Batch_size x num_classes] sized matrix
        x = elu(self.pool(self.conv1(x)))
        x = elu(self.conv2(x))
        x = elu(self.pool(self.conv3(x)))
        x = elu(self.conv4(x))
        # Output features of your second last layer
        if output_features: return x
        x = self.dropout(x)

        x = self.conv5(x)
        # Return the result of your network
        return torch.mean(x, dim=(2, 3))


def get_cifar10_dataset():
    data_transforms = Compose([Resize([64, 64]), ToTensor()])
    cifar_trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=data_transforms)
    cifar_testset = datasets.CIFAR10(root='.', train=False, download=True, transform=data_transforms)
    return cifar_trainset, cifar_testset



def get_oxford_dataset():
    data_transforms = Compose([Resize([64, 64]), ToTensor()])
    label_transforms = Compose([Resize([64, 64], InterpolationMode.NEAREST), PILToTensor()])

    label_map = ['cat', 'dog']
    breads_to_cat_or_dog = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                            1,
                            0, 0, 1, 1, 1]

    def transform_labels(inp):
        seg_img, label = inp
        return torch.minimum(label_transforms(seg_img) - 1, torch.tensor(1)).to(torch.float32), breads_to_cat_or_dog[
            label]

    train_data = datasets.OxfordIIITPet('.', 'trainval', target_types=('segmentation', 'category'),
                                        transform=data_transforms, download=True, target_transform=transform_labels)
    test_data = datasets.OxfordIIITPet('.', 'test', target_types=('segmentation', 'category'),
                                       transform=data_transforms,
                                       download=True, target_transform=transform_labels)
    return train_data, test_data


def train_simple_net_cifar():
    NUM_EPOCHS = 10
    train_data, test_data = get_cifar10_dataset()
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    print('seed', torch.seed())
    model = SimpleNet(num_classes=10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # , lr=0.0005)
    calculate_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        cnt = 0
        tic = time.time()

        if hasattr(model, "dropout"):
            print('dropout training', model.dropout.training)
            model.dropout.training = True

        for i, (img, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = calculate_loss(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss
            cnt += 1
            if i % 10 == 0:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f, time: %.3f' %
                      (epoch + 1, i + 1, running_loss / cnt, time.time() - tic))
                running_loss = 0.0
                cnt = 0
                tic = time.time()
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for inputs_test, (seg_mask, labels_test) in test_dataloader:
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                test_output = model(inputs_test)
                test_loss.append(calculate_loss(test_output, labels_test).cpu().numpy())
                test_acc.append((test_output.argmax(1) == labels_test).double().mean().cpu().numpy())
        print('TEST [%d, %5d] loss: %.3f, acc: %.3f' %
              (epoch + 1, i + 1, np.mean(test_loss), np.mean(test_acc)))


def train_simple_net_pets():
    NUM_EPOCHS = 10
    train_data, test_data = get_oxford_dataset()
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    print('seed', torch.seed())
    model = SimpleNet(num_classes=2).to(device)

    lr_map = {3: 0.0001}
    # lr_map = {1: 0.0002, 3: 0.0001}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # , lr=0.0005)
    calculate_loss = torch.nn.CrossEntropyLoss()
    print(len(train_data))
    acc_accum, loss_accum = run_test(model, test_dataloader, calculate_loss)
    print('TEST [%d, %5d] loss: %.3f, acc: %.3f' %
          (0, 0, loss_accum, acc_accum))
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        cnt = 0
        tic = time.time()
        if epoch in lr_map:
            for g in optimizer.param_groups:
                g['lr'] = lr_map[epoch]

        if hasattr(model, "dropout"):
            print('dropout training', model.dropout.training)
            model.dropout.training = True

        for i, (img, (seg_mask, label)) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = calculate_loss(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss
            cnt += 1
            if i % 10 == 0:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f, time: %.3f' %
                      (epoch + 1, i + 1, running_loss / cnt, time.time() - tic))
                running_loss = 0.0
                cnt = 0
                tic = time.time()
        acc_accum, loss_accum = run_test(model, test_dataloader, calculate_loss)
        print('TEST [%d, %5d] loss: %.3f, acc: %.3f' %
              (epoch + 1, i + 1, loss_accum, acc_accum))


def run_test(model, test_dataloader, calculate_loss):
    if hasattr(model, "dropout"):
        print('dropout training', model.dropout.training)
        model.dropout.training = False
    with torch.no_grad():
        loss_accum = 0
        acc_accum = 0
        cnt = 0
        for inputs_test, (seg_mask, labels_test) in test_dataloader:
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
            test_output = model(inputs_test)
            test_loss = calculate_loss(test_output, labels_test)
            test_acc = (test_output.argmax(1) == labels_test).double().mean()
            loss_accum += test_loss.detach()
            acc_accum += test_acc.detach()
            cnt += 1
    return acc_accum / cnt, loss_accum / cnt


if __name__ == '__main__':
    train_simple_net_pets()
    #train_simple_net_cifar()

#%%
