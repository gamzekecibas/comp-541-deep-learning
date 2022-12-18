
from CnnModel import SimpleCNN
from DataLoader import *
from torch import nn
from DataLoader import LoaderClass
import time
import torch
from torch.nn import CrossEntropyLoss
import tqdm
class Trainer():
    def __init__(self,model,criterion,tr_loader,val_loader,optimizer,num_epoch,patience,batch_size,lr_scheduler=None):
        self.model = model
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.softmax = nn.Softmax()
        self.no_inc = 0
        self.best_loss = 9999
        self.phases = ["train","val"]
        self.best_model = []
        self.best_val_acc = 0
        self.best_train_acc = 0
        self.best_val_loss = 0
        self.best_train_loss = 0
        self.batch_size = batch_size
        if self.model.use_cuda:
            self.model = self.model.cuda()


        pass
    def train(self):
        pbar = tqdm.tqdm(desc= "Epoch 0, phase: Train",postfix="train_loss : ?, train_acc: ?")
        for i in range(self.num_epoch):
            last_train_acc = 0
            last_val_acc = 0
            last_val_loss = 0
            last_train_loss = 0
            pbar.update(1)

            for phase in self.phases:
                total_acc = 0
                total_loss = 0
                start = time.time()
                if phase == "train":
                    pbar.set_description_str("Epoch %d,"% i + "phase: Training")
                    loader = self.tr_loader
                    self.model.train()
                else:
                    pbar.set_description_str("Epoch %d,"% i + "phase: Validation")
                    loader = self.val_loader
                    self.model.eval()
                iter = 0
                for images,labels in loader:
                    iter += 1
                    if self.model.use_cuda:

                        images = images.to(0)
                        labels = labels.to(0)
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    softmaxed_scores = self.softmax(logits)
                    _, predictions = torch.max(softmaxed_scores,1)
                    _, labels = torch.max(labels,1)
                    loss = self.criterion(softmaxed_scores.float(),labels.long())
                    total_loss += loss.item()
                    total_acc += torch.sum(predictions == labels).item()

                    if phase == "train":
                        pbar.set_postfix_str("train acc: %6.3f," %(total_acc/ (iter*self.batch_size)) + ("train loss: %6.3f" % (total_loss / iter)))
                        loss.backward()
                        self.optimizer.step()
                    else:
                        pbar.set_postfix_str("val acc: %6.3f," %(total_acc/ (iter*self.batch_size)) + ("val loss: %6.3f" % (total_loss / iter)))


                if phase == "train":
                    if self.lr_scheduler:

                        self.lr_scheduler.step()
                end = time.time()
                if phase == "train":
                    loss_p = total_loss / iter
                    acc_p = total_acc / len(self.tr_loader.dataset)
                    last_train_acc = acc_p
                    last_train_loss = loss_p
                else:
                    loss_p = total_loss / iter
                    acc_p = total_acc / len(self.val_loader.dataset)
                    last_val_acc = acc_p
                    last_val_loss = loss_p

                    if loss_p < self.best_loss:
                        print("New best loss, loss is: ",str(loss_p), "acc is: ",acc_p )
                        self.best_loss = loss_p
                        self.no_inc = 0
                        self.best_model = self.model
                        self.best_train_acc = last_train_acc
                        self.best_train_loss = last_train_loss
                        self.best_val_loss = last_val_loss
                        self.best_val_acc = last_val_acc
                    else:
                        print("Not a better score")


                        self.no_inc += 1
                        if self.no_inc == self.patience:
                            print("Out of patience returning the best model")
                            print(
                                "Best val acc: {}, Best val loss: {}, Best train acc: {}, Best train loss: {} ".format(
                                    self.best_val_acc, self.best_val_loss, self.best_train_acc, self.best_train_loss
                                ))  # Stats of the best model
                            return self.best_model
        print("Training ended returning the best model")
        print(
            "Best val acc: {}, Best val loss: {}, Best train acc: {}, Best train loss: {} ".format(
                self.best_val_acc, self.best_val_loss, self.best_train_acc, self.best_train_loss
            ))  # Stats of the best model
        return self.best_model