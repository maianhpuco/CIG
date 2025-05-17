import os
import sys 
import numpy as np
import logging
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim 
import yaml  

def get_region_original_size(slide, xywh_abs_bbox):
    xmin_original, ymin_original, width_original, height_original = xywh_abs_bbox
    region = slide.read_region(
        (xmin_original, ymin_original),  # Top-left corner (x, y)
        0,  # Level 0
        (width_original, height_original)  # Width and height
    )
    return region.convert('RGB')


def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
 

def read_region_from_npy(dir_folder, slide_basename, foreground_idx):
    # Construct the file path
    npy_file_path = os.path.join(dir_folder, slide_basename, f"{foreground_idx}.npy")

    # Check if the file exists
    if not os.path.exists(npy_file_path):
        raise FileNotFoundError(f"The file {npy_file_path} does not exist.")

    # Load and return the NumPy array from the file
    region_np = np.load(npy_file_path)

    return region_np 

# Function to set up the logger
def setup_logger(log_file="training_log.txt"):
    log_dir = os.path.dirname(log_file)
    
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create directory if it doesn't exist 
    print("completed create a log file at: ", log_dir)
     
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the log level to INFO to capture all log messages
    
    # Create a file handler to log to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 

# Define the simple model
# class SimpleModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
    
#     def forward(self, x):
#         return self.fc(x)

# Example function to simulate training and validation
# def train_and_validate(logger):
#     # Simple dataset
#     inputs = torch.randn(100, 10)
#     labels = torch.randint(0, 2, (100,))

#     dataset = TensorDataset(inputs, labels)
#     train_loader = DataLoader(dataset, batch_size=10)
    
#     # Model, Loss and Optimizer
#     model = SimpleModel(10, 2)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)
    
#     # Training loop
#     for epoch in range(5):  # 5 epochs
#         model.train()  # Set the model to training mode
#         train_loss = 0.0
#         train_error = 0.0
        
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()
            
#             # Forward pass
#             output = model(data)
#             loss = loss_fn(output, target)
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
            
#             # Calculate error (just a simple example of using loss)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct = pred.eq(target.view_as(pred)).sum().item()
#             train_loss += loss.item()
#             train_error += (data.size(0) - correct)  # Count the incorrect predictions
            
#             if batch_idx % 5 == 0:  # Log every 5 batches
#                 logger.info(f"Epoch {epoch+1} Batch {batch_idx} - Loss: {loss.item():.4f}, Train Error: {train_error / (batch_idx+1):.4f}")
        
#         # Log the epoch results
#         train_loss /= len(train_loader)
#         train_error /= len(train_loader.dataset)
#         logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f}")
        
#         # Validation (here, we use the same data for simplicity)
#         model.eval()  # Set the model to evaluation mode
#         val_loss = 0.0
#         val_error = 0.0
        
#         with torch.no_grad():
#             for data, target in train_loader:
#                 output = model(data)
#                 loss = loss_fn(output, target)
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct = pred.eq(target.view_as(pred)).sum().item()
                
#                 val_loss += loss.item()
#                 val_error += (data.size(0) - correct)
        
#         val_loss /= len(train_loader)
#         val_error /= len(train_loader.dataset)
#         logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Error: {val_error:.4f}")
        
#         logger.info("-" * 40)

# if __name__ == "__main__":
    # Set up logger
    # logger = setup_logger("./logs/training_log.txt")
    
    # Run training and validation
    # train_and_validate(logger) 