# encoding_size = 1024
# settings = {'num_splits': args.k, 
#             'k_start': args.k_start,
#             'k_end': args.k_end,
#             'task': args.task,
#             'max_epochs': args.max_epochs, 
#             'results_dir': args.results_dir, 
#             'lr': args.lr,
#             'experiment': args.exp_code,
#             'reg': args.reg,
#             'label_frac': args.label_frac,
#             'bag_loss': args.bag_loss,
#             'seed': args.seed,
#             'model_type': args.model_type,
#             'model_size': args.model_size,
#             "use_drop_out": args.drop_out,
#             'weighted_sample': args.weighted_sample,
#             'opt': args.opt}

# if args.model_type in ['clam_sb', 'clam_mb']:
#    settings.update({'bag_weight': args.bag_weight,
#                     'inst_loss': args.inst_loss,
#                     'B': args.B}) 

# TODO
# [ ] adding the Early Stoping
# [ ] adjusting the train_one_epoch
# [ ] adjusting the train 

import torch
import numpy as np
import torch
from utils.utils import *
import os
# from dataset_modules.dataset_generic import save_splits
# from models.model_mil import MIL_fc, MIL_fc_mc
# from models.model_clam import CLAM_MB, CLAM_SB
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

def temp_train_loop(features, label, model, optimizer, n_classes, bag_weight, loss_fn=None, device=None):
    logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True) 
    loss = loss_fn(logits, label)
    loss_value = loss.item() 
    
    instance_loss = instance_dict['instance_loss']
    # inst_count+=1
    instance_loss_value = instance_loss.item()
    # train_inst_loss += instance_loss_value
    
    total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

    inst_preds = instance_dict['inst_preds']
    inst_labels = instance_dict['inst_labels'] 
    # train_loss += loss_value
    print('loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(loss_value, instance_loss_value, total_loss.item()) + 
        'label: {}, bag_size: {}'.format(label.item(), features.size(0)))

    error = calculate_error(Y_hat, label)
    print("error", error)
    
def train_all_epochs(datasets, cur, logger):
    """
    Train the model for a single fold across multiple epochs
    """
    # print(f'\nTraining Fold {cur}!')
    # logger = setup_logger('./logs/test_clam.txt')  # Set up logger to record output to a file

    # # Initialize train, validation, and test splits
    # print('\nInit train/val/test splits...', end=' ')
    # train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, f'splits_{cur}.csv'))
    # print('Done!')

    # # Display number of samples
    # logger.info(f"Training on {len(train_split)} samples")
    # logger.info(f"Validating on {len(val_split)} samples")
    # logger.info(f"Testing on {len(test_split)} samples")

    # # Initialize loss function
    # logger.info('\nInit loss function...')
    # loss_fn = get_loss_function(args)
    # logger.info('Done!')

    # # Initialize model
    # logger.info('\nInit Model...')
    # model = initialize_model(args)
    # model.to(device)
    # logger.info('Done!')
    # print_network(model)

    # # Set up optimizer
    # logger.info('\nInit optimizer ...')
    # optimizer = get_optim(model, args)
    # logger.info('Done!')

    # # Initialize data loaders
    # logger.info('\nInit Loaders...')
    # train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    # val_loader = get_split_loader(val_split, testing=args.testing)
    # test_loader = get_split_loader(test_split, testing=args.testing)
    # logger.info('Done!')

    # # Setup early stopping
    # logger.info('\nSetup EarlyStopping...')
    early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    logger.info('Done!')

    # Run the training loop across all epochs
    for epoch in range(args.max_epochs):
        stop = train_epoch(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, logger, loss_fn)

        if stop:
            break

        # Validate the model after every epoch
        
        # stop = validate_epoch(cur, epoch, model, val_loader, args.n_classes, early_stopping, logger, loss_fn, args.results_dir)

        # if stop:
        #     break

    # Save the final model state
    # if args.early_stopping:
    #     model.load_state_dict(torch.load(os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))

    # # Print the validation results
    # val_error, val_auc = print_results(model, val_loader, args.n_classes)
    # logger.info(f'Val error: {val_error:.4f}, ROC AUC: {val_auc:.4f}')

    # # Print the test results
    # test_error, test_auc = print_results(model, test_loader, args.n_classes)
    # logger.info(f'Test error: {test_error:.4f}, ROC AUC: {test_auc:.4f}')

    return train_acc #, test_auc, val_auc
 
          
def train_epoch(
    epoch, 
    model, 
    dataset, 
    optimizer, 
    n_classes, 
    bag_weight, 
    logger=None, 
    loss_fn=None
):
    # Set model to training mode
    model.train()
    
    # Initialize accuracy loggers for both overall and instance-level accuracy
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    # Initialize loss and error accumulators
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # Log header for the epoch
    logger.info(f"Starting epoch {epoch}...")
    
    # Move data to device (GPU or CPU)
    for data in dataset: 
        features, label, patch_indices, coordinates, spixels_indices, file_basename= data    

        label = label.long()
        # print("features", features.shape)
        # print("indices", patch_indices)
        # print("label shape: ", label.shape) 
        
        features, label = features.to(device), label[0].to(device)

        # Perform forward pass through the model
        logits, Y_prob, Y_hat, _, instance_dict = model(
            features, label=label, instance_eval=True)

        # Log overall accuracy
        acc_logger.log(Y_hat, label)
        
        # Calculate loss
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        # Instance-level loss
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        # Total loss is a weighted combination of the bag-level and instance-level losses
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss 
       

        # print(f"bag loss {loss.item()}, instance loss {instance_loss.item()}, total loss {total_loss}")
        # train_losses.append(total_loss.item())
        
        # Log instance-level accuracy
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        
        inst_logger.log_batch(inst_preds, inst_labels)

        # Accumulate the loss
        train_loss += loss_value

        # Calculate training error
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # Perform backward pass and optimizer step
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Calculate and log the average loss and error for the epoch
    train_loss /= len(dataset)  # Since we're not dealing with batches, we just use the single example
    train_error /= len(dataset)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        # Log instance accuracy for each class
        for i in range(n_classes):
            acc, correct, count = inst_logger.get_summary(i)
            logger.info(f"Class {i} Clustering Accuracy: {acc}, Correct: {correct}/{count}")

    logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Clustering Loss: {train_inst_loss:.4f}, Train Error: {train_error:.4f}")

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logger.info(f"Class {i}: Accuracy: {acc}, Correct: {correct}/{count}")
    
    return train_loss 


def eval(
    epoch, 
    model, 
    dataset, 
    n_classes, 
    bag_weight, 
    logger=None, 
    loss_fn=None
):
    # Set model to training mode
    model.eval()
    
    # Initialize accuracy loggers for both overall and instance-level accuracy
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    # Initialize loss and error accumulators
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # Log header for the epoch
    logger.info(f"Starting epoch {epoch}...")
    
    # Move data to device (GPU or CPU)
    for data in dataset: 
        features, label, patch_indices, coordinates, spixels_indices, file_basename= data    

        label = label.long()
        # print("features", features.shape)
        # print("indices", patch_indices)
        # print("label shape: ", label.shape) 
        
        features, label = features.to(device), label[0].to(device)

        # Perform forward pass through the model
        logits, Y_prob, Y_hat, _, instance_dict = model(
            features, label=label, instance_eval=True)

        # Log overall accuracy
        acc_logger.log(Y_hat, label)
        
        # Calculate loss
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        # Instance-level loss
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        # Total loss is a weighted combination of the bag-level and instance-level losses
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss 
       

        # print(f"bag loss {loss.item()}, instance loss {instance_loss.item()}, total loss {total_loss}")
        # train_losses.append(total_loss.item())
        
        # Log instance-level accuracy
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        
        inst_logger.log_batch(inst_preds, inst_labels)

        # Accumulate the loss
        train_loss += loss_value

        # Calculate training error
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # # Perform backward pass and optimizer step
        # total_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
    
    # Calculate and log the average loss and error for the epoch
    train_loss /= len(dataset)  # Since we're not dealing with batches, we just use the single example
    train_error /= len(dataset)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        # Log instance accuracy for each class
        for i in range(n_classes):
            acc, correct, count = inst_logger.get_summary(i)
            # logger.info(f"Class {i} Clustering Accuracy: {acc}, Correct: {correct}/{count}")

    logger.info(f"Loss: {train_loss:.4f}, Clustering Loss: {train_inst_loss:.4f}, Train Error: {train_error:.4f}")

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logger.info(f"Class {i}: Accuracy: {acc}, Correct: {correct}/{count}")
    
    

    # print("train loss:", train_losses)

# if __name__=='__main__':
#     logger = setup_logger("./logs/training_log.txt")
#     epoch = 0
#     model = CLAM_SB()
#     train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, logger, loss_fn)
 
 

