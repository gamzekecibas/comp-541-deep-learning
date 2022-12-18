from CnnModel import SimpleCNN
from train import Trainer
from DataLoader import LoaderClass
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision import transforms, utils
import numpy as np
import os
from PIL import Image,ImageFile

LR = 1e-4
Momentum = 0.9 # If you use SGD with momentum
BATCH_SIZE = 16
USE_CUDA = False
POOLING = False
NUM_EPOCHS = 200
PATIENCE = -1
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.2
NUM_ARTISTS = 11
DATA_PATH = "./art_data/artists"
ImageFile.LOAD_TRUNCATED_IMAGES = True # Do not change this


def load_artist_data():
    data = []
    labels = []
    artists = [x for x in os.listdir(DATA_PATH) if x != '.DS_Store']
    print(artists)
    for folder in artists:
        class_index = artists.index(folder)
        for image_name in os.listdir(DATA_PATH + "/" + folder):
            img = Image.open(DATA_PATH + "/" + folder + "/" + image_name)
            artist_label = (np.arange(NUM_ARTISTS) == class_index).astype(np.float32)
            data.append(np.array(img))
            labels.append(artist_label)
    shuffler = np.random.permutation(len(labels))
    data = np.array(data)[shuffler]
    labels = np.array(labels)[shuffler]

    length = len(data)
    val_size = int(length*0.2)
    val_data = data[0:val_size+1]
    train_data = data[val_size+1::]
    val_labels = labels[0:val_size+1]
    train_labels = labels[val_size+1::]
    print(val_labels)
    data_dict = {"train_data":train_data,"val_data":val_data}
    label_dict = {"train_labels":np.array(train_labels),"val_labels":np.array(val_labels)}

    return data_dict,label_dict

if __name__ == "__main__":
    data,labels = load_artist_data()
    model = SimpleCNN(use_cuda=True,pooling=False)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    transforms = {
        'train': transforms.Compose([
            transforms.Resize(50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    train_dataset = LoaderClass(data,labels,"train",transforms["train"])
    valid_dataset = LoaderClass(data,labels,"val",transforms["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4, pin_memory=True)

    criterion = CrossEntropyLoss()
    trainer_m = Trainer(model, criterion, train_loader, val_loader, optimizer, num_epoch=NUM_EPOCHS, patience=PATIENCE,batch_size=BATCH_SIZE,lr_scheduler= None)
    best_model = trainer_m.train()