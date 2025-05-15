import torch
import numpy as np
import torch
from utils.utils import *
import os
import torch.nn.functional as F 
from src.bag_classifier.clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
 

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error 

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}") 
    
    return model, optimizer, epoch, loss 

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss

def train_epoch(
    epoch, 
    model, 
    dataset, 
    optimizer, 
    n_classes, 
    logger=None, 
    loss_fn=None, 
    checkpoint_filename='checkpoint.pt',
    save_last_epoch_checkpoint=True, 
):
    model.train()
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    for data in dataset: 
        features, label, patch_indices, coordinates, spixels_indices, file_basename = data    

        label = label.long()
        
        features, label = features.to(device), label[0].to(device)

        logits = model(features, label=label)
        Y_prob = F.softmax(logits, dim = 1) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]   
        
        loss = loss_fn(logits, label)
            
        acc_logger.log(Y_hat, label) 
        
        loss_value = loss.item()

        total_loss = loss 
        
        # No instance loss, just use bag-level loss
        train_loss += loss_value
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= len(dataset)  # Since we're not dealing with batches, we just use the single example

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logger.info(f"Epoch {epoch}, Class {i}: Accuracy: {acc}, Correct: {correct}/{count}")
    
    # logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f}")
    if save_last_epoch_checkpoint:
        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_filename)  
    return train_loss 



def eval(
    model, 
    optimizer, 
    dataset, 
    n_classes, 
    logger, 
    loss_fn, 
    checkpoint_filename=None, 
):
    if checkpoint_filename is not None: 
        try:
            model, optimizer, start_epoch, last_loss = load_checkpoint(model, optimizer, filename=checkpoint_filename)
            print(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting training from scratch") 
        
        
    model.eval()  # Set model to evaluation mode
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for data in dataset: 
            features, label, patch_indices, coordinates, spixels_indices, file_basename= data   
        
            label = label.long()
            
            features, label = features.to(device), label[0].to(device)

            # Forward pass (no backward pass for evaluation)
            logits = model(features, label=label)
            Y_prob = F.softmax(logits, dim = 1) 
            Y_hat = torch.topk(logits, 1, dim = 1)[1]   
            
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)
                
            acc_logger.log(Y_hat, label)  
            
            loss_value = loss.item()

            # Accumulate the loss and error
            val_loss += loss_value

    # Calculate the average loss and error for the entire dataset
    val_loss /= len(dataset)

    # Log the per-class accuracy and other metrics
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logger.info(f"Class {i}: Accuracy: {acc:.3f}, Correct: {correct}/{count}")

    logger.info(f" Validation Loss: {val_loss:.4f}")

    return val_loss
 