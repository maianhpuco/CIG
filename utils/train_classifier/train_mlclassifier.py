import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import torch.nn as nn 

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 
import pandas as pd 
def save_checkpoint(model, optimizer, epoch, best_auc, checkpoint_path="best_mil_model.pth"):
    """
    Saves a model checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auc": best_auc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model Checkpoint Saved: {checkpoint_path} (Best AUC: {best_auc:.4f})")

def load_checkpoint(model, optimizer, checkpoint_path="best_mil_model.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a model checkpoint for inference or resuming training.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_auc = checkpoint.get("best_auc", 0.0)  # Fix: Default to 0.0 if missing
        print(f"Loaded model checkpoint from {checkpoint_path} (Epoch {epoch}, Best AUC: {best_auc:.4f})")
        return model, optimizer, epoch, best_auc
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh training.")
        return model, optimizer, 0, 0.0  # Start from scratch


def train_mil_classifier(
    model,
    train_dataset,
    test_dataset,
    num_epochs=10,
    batch_size=32,
    gamma=1,
    alpha=0.8,
    lr=0.001,
    weight_decay=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    scheduler_patience=3,
    checkpoint_path="best_mil_model.pth",
    allow_load_checkpoint=False
):
    """
    Trains a Multiple Instance Learning (MIL) classifier using Focal Loss.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=scheduler_patience, factor=0.5, verbose=True
    )

    # Compute class weights dynamically
    labels = [int(sample["label"].item()) for sample in train_dataset]  # Fix: Ensure integer labels
    normal_count = sum(1 for l in labels if l == 0)
    tumor_count = sum(1 for l in labels if l == 1)

    print(f"Class Distribution: Normal={normal_count}, Tumor={tumor_count}")

    # Create weighted sampling
    class_counts = [normal_count, tumor_count]
    weights = [1.0 / class_counts[label] for label in labels]  # Fix: Assign weights correctly
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_mil_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mil_fn)

    # Load checkpoint if exists
    best_auc = 0.0
    start_epoch = 0
    if allow_load_checkpoint:
        model, optimizer, start_epoch, best_auc = load_checkpoint(model, optimizer, checkpoint_path, device)

    # Instantiate Focal Loss once
    criterion = FocalLoss(gamma=gamma, alpha=alpha)

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in tqdm_bar:
            features, bag_labels, bag_sizes = (
                batch["features"].to(device),
                batch["label"].to(device),
                batch["bag_sizes"],
            )

            bag_labels = bag_labels.view(-1, 1)  # Ensure shape is (batch_size, 1)

            # Forward pass (get **logits**, no Sigmoid applied)
            bag_outputs = model(features, bag_sizes).view(-1, 1)  # Shape: (batch_size, 1)

            # Compute Focal Loss directly on logits (NOT after sigmoid)
            loss = criterion(bag_outputs, bag_labels.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Compute accuracy (convert logits to probability)
            preds = (torch.sigmoid(bag_outputs) > 0.5).long()
            correct += (preds == bag_labels.long()).sum().item()
            total += bag_labels.numel()
            total_loss += loss.item()

            tqdm_bar.set_postfix(loss=loss.item(), acc=correct / total)

        # Compute Training Metrics
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}")

        # Evaluate Model
        test_loss, test_acc, test_auc = evaluate_mil_classifier(model, test_loader, criterion, device)

        # Adjust Learning Rate Based on AUC
        scheduler.step(test_auc)

        # Save Best Model Based on AUC Score
        if test_auc > best_auc:
            best_auc = test_auc
            save_checkpoint(model, optimizer, epoch, best_auc, checkpoint_path)

        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in MIL.
    """
    def __init__(self, gamma=1, alpha=0.9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()  # Average loss for batch

def find_best_threshold(all_labels, all_probs):
    """
    Finds the best decision threshold for classification using Precision-Recall Curve.
    """
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    if len(thresholds) == 0:  #Fix: Handle edge case (single-class problem)
        return 0.5  # Default threshold if PR curve is undefined

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid div by zero
    best_threshold = thresholds[np.argmax(f1_scores)]  # Choose threshold with best F1
    return best_threshold

def evaluate_mil_classifier(model, test_loader, criterion, device):
    """
    Evaluates the MIL model on the test dataset.

    Returns:
        tuple: (test_loss, test_accuracy, auc_score)
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    correct_class_0 = 0
    correct_class_1 = 0
    total_class_0 = 0
    total_class_1 = 0

    with torch.no_grad():
        for batch in test_loader:
            features, bag_labels, bag_sizes = (
                batch["features"].to(device),
                batch["label"].to(device),
                batch["bag_sizes"]
            )

            bag_labels = bag_labels.view(-1, 1)  # Reshape labels
            bag_outputs = model(features, bag_sizes).view(-1, 1)  # Get logits

            loss = criterion(bag_outputs, bag_labels.float())  # Compute loss

            # Convert logits to probabilities
            probs = torch.sigmoid(bag_outputs)

            # Collect labels & probabilities for threshold tuning
            all_labels.append(bag_labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            total_loss += loss.item()

    # Convert to NumPy efficiently
    if len(all_labels) == 0:
        print("⚠ Warning: No test samples were processed.")
        return float("nan"), float("nan"), float("nan")

    all_labels = np.vstack(all_labels).flatten()
    all_probs = np.vstack(all_probs).flatten()

    # Compute AUC safely
    if len(np.unique(all_labels)) > 1:
        auc_score = roc_auc_score(all_labels, all_probs)
    else:
        auc_score = 0.5  # Default AUC if only one class exists

    # Find optimal threshold
    best_threshold = find_best_threshold(all_labels, all_probs)

    # Apply threshold to predictions
    preds = (all_probs > best_threshold).astype(int)

    # Compute accuracy per class
    correct_class_0 = np.sum((preds == 0) & (all_labels == 0))
    correct_class_1 = np.sum((preds == 1) & (all_labels == 1))
    total_class_0 = np.sum(all_labels == 0)
    total_class_1 = np.sum(all_labels == 1)

    acc_class_0 = correct_class_0 / total_class_0 if total_class_0 > 0 else 0
    acc_class_1 = correct_class_1 / total_class_1 if total_class_1 > 0 else 0
    overall_acc = (correct_class_0 + correct_class_1) / len(all_labels) if len(all_labels) > 0 else 0
    print("---- Evaluation result: ")
    print(f"Test Loss = {total_loss/len(test_loader):.4f}, Test Accuracy = {overall_acc:.4f}")
    print(f"AUC = {auc_score:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Class 0: {correct_class_0}/{total_class_0} correct ({acc_class_0:.4f} accuracy)")
    print(f"Class 1: {correct_class_1}/{total_class_1} correct ({acc_class_1:.4f} accuracy)")
    print("-------")

    return total_loss / len(test_loader), overall_acc, auc_score


def predict_and_save(model, test_dataset, criterion, device, output_file="predictions.csv"):
    """
    Evaluates the MIL model on the dataset (without using DataLoader) and saves predictions.

    Args:
        model (torch.nn.Module): The trained MIL model.
        test_dataset (Dataset): The dataset to evaluate.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Computation device.
        output_file (str): Path to save CSV predictions.

    Returns:
        tuple: (test_loss, test_accuracy, auc_score)
    """
    model.eval()
    total_loss = 0
    all_file_basenames = []
    all_labels = []
    all_logits = []
    all_probs = []

    with torch.no_grad():
        for sample in test_dataset:
            features = sample["features"].to(device)
            bag_label = torch.tensor([sample["label"]], dtype=torch.float32).view(-1, 1).to(device)  # Fix shape mismatch
            bag_size = [features.shape[0]]  # Single bag

            # File name
            file_basename = sample["file_basename"]

            # Forward pass
            bag_output = model(features, bag_size).view(-1, 1)  # Get logits

            loss = criterion(bag_output, bag_label)  # Compute loss

            # Convert logits to probabilities
            prob = torch.sigmoid(bag_output)

            # Store results
            all_file_basenames.append(file_basename)
            all_labels.append(bag_label.cpu().numpy())
            all_logits.append(bag_output.cpu().numpy())
            all_probs.append(prob.cpu().numpy())

            total_loss += loss.item()

    # Convert to NumPy efficiently
    if len(all_labels) == 0:
        print("⚠ Warning: No samples were processed.")
        return float("nan"), float("nan"), float("nan")

    all_labels = np.vstack(all_labels).flatten()
    all_logits = np.vstack(all_logits).flatten()
    all_probs = np.vstack(all_probs).flatten()

    # # Compute AUC safely
    # if len(np.unique(all_labels)) > 1:
    #     auc_score = roc_auc_score(all_labels, all_probs)
    # else:
    #     auc_score = 0.5  # Default AUC if only one class exists

    # Find optimal threshold
    best_threshold = find_best_threshold(all_labels, all_probs)

    # Apply threshold to predictions
    preds = (all_probs > best_threshold).astype(int)

    # Save predictions
    df = pd.DataFrame({
        "file_basename": all_file_basenames,
        "logit": all_logits,
        "probability": all_probs,
        "predicted_label": preds,
        "true_label": all_labels, 
        "best_threshold": best_threshold, 
        "correct": (preds == all_labels).astype(int)
    })
    
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # return total_loss / len(test_dataset), (preds == all_labels).mean(), auc_score
 
def collate_mil_fn(batch):
    """
    Collates a batch of samples into a batch dictionary for Multiple Instance Learning.

    Args:
        batch (list): List of samples, each containing instances and a bag label.

    Returns:
        dict: Dictionary containing batched tensors.
    """
    features_list = []
    labels_list = []
    bag_sizes = []
    file_basenames = []
    
    for item in batch:
        features = item["features"]  # Shape: (num_instances, feature_dim)
        label = item["label"].item()  # Convert to scalar
        
        bag_sizes.append(features.shape[0])  # Store instance count per bag

        features_list.append(features)
        labels_list.append(torch.tensor([label], dtype=torch.float32))  # Store bag label
        # file_basenames.append(item["file_basename"]) 
        
    # Stack features and labels
    features = torch.cat(features_list, dim=0)  # Shape: (total_instances, feature_dim)
    labels = torch.cat(labels_list, dim=0)  # Shape: (batch_size, 1)

    return {
        "features": features,  # Shape: (total_instances, feature_dim)
        "label": labels,  # Shape: (batch_size, 1)
        "bag_sizes": bag_sizes,  # List of instance counts per bag
        # "file_basename": file_basenames
    }
