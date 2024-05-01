import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io
import matplotlib.pyplot as plt
import random
from skimage import data
from scipy.ndimage import rotate
from kernels import *
import os
from torch.utils.data import Subset
from collections import Counter
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import seaborn as sns  
import numpy as np
from collections import defaultdict


from PIL import Image, ImageFilter
import io
import re
import random
import numpy.random as npr
from skimage import data
from scipy.ndimage import rotate
from kernels import *
import torchvision
import os
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as transforms
 
import my_utils as ut
from transformers import Swinv2ForImageClassification, SwinConfig
from torch.optim import AdamW
from torchvision import transforms, datasets
import math

class DatasetAI(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.samples = []
        self.file_paths = []  # Store file paths for checking duplicates


        for model_name in sorted(os.listdir(root_dir)):
            model_path = os.path.join(root_dir, model_name)
            if os.path.isdir(model_path):
                imagenet_dir = f'imagenet_{model_name}'
                data_dir = os.path.join(model_path, imagenet_dir, split)
        
                
                if os.path.isdir(data_dir):
                    for class_label in ['ai', 'nature']:
                        class_path = os.path.join(data_dir, class_label)
                        if os.path.exists(class_path):
                            for img_name in os.listdir(class_path):
                                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(class_path, img_name)
                                    self.samples.append((img_path, class_label, model_name))
                                    self.file_paths.append(img_path)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        img_path, class_label, model_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        rich, poor = smash_n_reconstruct(image)  # Ensure this function doesn't cause data modification
        
        if self.transform:
            rich = self.transform(rich)
            poor = self.transform(poor)
        
        label = 0 if class_label == 'ai' else 1
        return rich, poor, label, model_name



class DeepClassifier(nn.Module):
    def __init__(self, num_classes= 2): 
        super(DeepClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(32)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.avg_pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.avg_pool(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.avg_pool(x)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.adaptive_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class HighPassFilters(nn.Module):
    def __init__(self, kernels):
        super(HighPassFilters, self).__init__()
        # Kernels are a parameter but not trained
        self.kernels = nn.Parameter(kernels, requires_grad=False)

    def forward(self, x):
        # Apply convolution with padding to maintain output size equal to input size
        return F.conv2d(x, self.kernels, padding =2)  # Padding set to 2 to maintain output size


class CNNBlock(nn.Module):
   def __init__(self, kernals):
       super(CNNBlock, self).__init__()
       self.conv = nn.Conv2d(30, 3, kernel_size=1,padding=0)
       self.filters = HighPassFilters(kernals)
       self.bn = nn.BatchNorm2d(3)
       self.htanh = nn.Hardtanh()
   def forward(self, x):
       x = self.filters(x)
       x = self.conv(x)
       x = self.bn(x)
       x = self.htanh(x)
       return x
  
def train_and_validate(model, train_loader, valid_loader, optimizer, device, num_epochs, best_model_path):
    criterion = nn.CrossEntropyLoss()
    try:
        checkpoint = torch.load(best_model_path)
        best_val_accuracy = checkpoint['best_val_accuracy']
        print("Loaded previous best model with accuracy:", best_val_accuracy)
    except FileNotFoundError:
        best_val_accuracy = float('-inf')
        print("No saved model found. Starting fresh!")

    for epoch in range(num_epochs):
        # # Training Phase
        model.train()
        total_train_loss, total_train, correct_train = 0, 0, 0
        for batch in train_loader:
            rich, poor, labels, model_names = batch  # Unpack model_names as well
            rich = rich.to(device)
            poor = poor.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(rich, poor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = total_train_loss / total_train
        train_accuracy = correct_train / total_train

        # Validation Phase
        model.eval()
        total_val_loss, total_val, correct_val = 0, 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                rich, poor, labels, model_names = batch  # Unpack model_names as well
                rich = rich.to(device)
                poor = poor.to(device)
                labels = labels.to(device)

                outputs = model(rich, poor)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = total_val_loss / total_val
        val_accuracy = correct_val / total_val

        # Print overall validation accuracy
        print(f'Epoch {epoch+1}/{num_epochs}\n,'
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n')
        
        # Check if general accuracy is the best and save
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({'model_state': model.state_dict(),
                        'best_val_accuracy': best_val_accuracy},
                       best_model_path)
            print(f"Saved new best model with accuracy: {best_val_accuracy:.4f}")
def test(model, test_loader, device, weights_path,  name_model=None):
    try:
        # Load the best model
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()
    seen_models = ["ADM","BigGAN","glide", "Midjourney","stable_diffusion_v_1_4","stable_diffusion_v_1_5", "VQDM","wukong"]
    all_labels = []
    all_predictions = []
    per_model_labels = {}
    per_model_predictions = {}
    seen_labels = []
    seen_predictions = []
    unseen_labels = []
    unseen_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            rich, poor, labels, model_names = batch
            rich = rich.to(device)
            poor = poor.to(device)
            labels = labels.to(device)
            
            outputs = model(rich, poor)
           
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for model_name, label, prediction in zip(model_names, labels, predicted):
                if model_name in seen_models:
                    seen_labels.append(label.item())
                    seen_predictions.append(prediction.item())
                else:
                    unseen_labels.append(label.item())
                    unseen_predictions.append(prediction.item())

                if model_name not in per_model_labels:
                    per_model_labels[model_name] = []
                    per_model_predictions[model_name] = []
                per_model_labels[model_name].append(label.item())
                per_model_predictions[model_name].append(prediction.item())
        
                

    display_confusion_matrices(per_model_labels, per_model_predictions, np.array(all_labels), np.array(all_predictions), np.array(seen_labels), np.array(seen_predictions), np.array(unseen_labels), np.array(unseen_predictions), name_model)

def display_confusion_matrices(per_model_labels, per_model_predictions, all_labels, all_predictions, seen_labels, seen_predictions, unseen_labels, unseen_predictions, name_model):
    metrics_data = []
    for model_name, labels in per_model_labels.items():
        predictions = per_model_predictions[model_name]
        cm = confusion_matrix(labels, predictions)
        TN = cm[0, 0] if cm.shape[0] > 1 else 0
        FP = cm[0, 1] if cm.shape[0] > 1 else 0
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        metrics_data.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "TN (Only AI Generated Image)": TN, 
            "FP (Only AI Generated Image)": FP,  
        })

    metrics_df = pd.DataFrame(metrics_data)
    
    # Creating a larger figure to accommodate both plots and table
    fig, axs = plt.subplots(2, 3, figsize=(24, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{name_model} Confusion Matrices and Metrics', fontsize=16)
    for ax in axs[0, :]:
        ax.set_aspect('equal')  # Ensures each cell of the heatmap is squar
    # Plot confusion matrices
    plot_confusion_matrix(axs[0, 0], all_labels, all_predictions, 'Overall')
    plot_confusion_matrix(axs[0, 1], seen_labels, seen_predictions, 'Seen Models')
    plot_confusion_matrix(axs[0, 2], unseen_labels, unseen_predictions, 'Unseen Models')
     # Adjust spacing between subplots
    plt.subplots_adjust(wspace=.1, hspace=.1)

    # Hide axes for the lower row used for the table
    for ax in axs[1, :]:
        ax.axis('off')

    # Placing the table in the middle lower grid and adding a title
    axs[1, 1].axis('on')
    axs[1, 1].axis('tight')
    axs[1, 1].axis('off')
    the_table = axs[1, 1].table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    axs[1, 1].set_title('Metrics Per Model', pad=10, fontsize=14)  # Adjust 'pad' as needed
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    the_table.auto_set_column_width(list(range(len(metrics_df.columns))))  # Adjust column widths
    
    # Make column headers bold
    for (i, key) in enumerate(metrics_df.columns):
        the_table[(0, i)].get_text().set_weight('bold')
        the_table[(0, i)].set_facecolor('#D3D3D3')  # You can change the color

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.save(f'/home/kosta/code/School/SentryAI/pth/Metrics{name_model}_confusion_matrix.png')
def plot_confusion_matrix(ax, labels, predictions, title):
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Purples', ax=ax, cbar= False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Calculate cell text colors based on background color
    threshold = cm.max() / 2.

    # Place text in the center of each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{get_label(i, j, labels, predictions)}\n({cm[i, j]})", 
                    ha="center", va="center", color="white" if cm[i, j] > threshold else "black", fontweight="bold")

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    ax.set_xlabel(f'Predicted\n\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}')

def get_label(i, j, labels, predictions):
    if i == j:
        return 'TP' if i == 1 else 'TN'
    else:
        return 'FP' if i < j else 'FN'

def split_datasets(train_dataset, val_test_dataset, train_size, val_size, test_size, seed_train=42, seed_val=42, seed_test=42):
    rngTrain = np.random.default_rng(seed_train)
    rngVal = np.random.default_rng(seed_val)
    rngTest = np.random.default_rng(seed_test)  # seed_test can be None for true randomness each time

    indices_dict = defaultdict(lambda: {'train': [], 'val': []})
   
    # Collect indices for training and validation datasets
    for idx, (_, class_label, model_name) in enumerate(train_dataset.samples):
        indices_dict[model_name]['train'].append(idx)
    
    for idx, (_, class_label, model_name) in enumerate(val_test_dataset.samples):
        indices_dict[model_name]['val'].append(idx)

    num_models = len(indices_dict)
    model_train_size = train_size // num_models
    model_val_size = val_size // num_models
    model_test_size = test_size // num_models

    aggregated_train_indices = []
    aggregated_val_indices = []
    aggregated_test_indices = []

    for model_name, indices in indices_dict.items():
        train_indices = np.array(indices['train'])
        val_indices = np.array(indices['val'])

        rngTrain.shuffle(train_indices)
        rngVal.shuffle(val_indices)

        # Balance training indices
        if len(train_indices) >= model_train_size:
            train_balanced_indices = rngTrain.choice(train_indices, size=model_train_size, replace=False)
            aggregated_train_indices.extend(train_balanced_indices)

        # Allocate indices for validation and test sets
        if len(val_indices) >= model_val_size + model_test_size:
            val_balanced_indices = val_indices[:model_val_size]
            test_balanced_indices = val_indices[model_val_size:model_val_size + model_test_size]
        else:
            split_index = int(len(val_indices) * (model_val_size / (model_val_size + model_test_size)))
            val_balanced_indices = val_indices[:split_index]
            test_balanced_indices = val_indices[split_index:split_index + model_test_size]

        aggregated_val_indices.extend(val_balanced_indices)
        rngTest.shuffle(test_balanced_indices)  # Shuffle for randomness if seed_test is None
        aggregated_test_indices.extend(test_balanced_indices)

    # Create subsets
    train_subset = Subset(train_dataset, aggregated_train_indices)
    val_subset = Subset(val_test_dataset, aggregated_val_indices)
    test_subset = Subset(val_test_dataset, aggregated_test_indices)
    
    return train_subset, val_subset, test_subset
def img_to_patches(img, min_patches=128, patch_size=32) -> tuple:

  
    # Calculate the number of patches for a 256x256 image
    target_patches = (256 // patch_size) ** 2

    # Calculate the number of patches for the current image
    current_patches_x = img.size[0] // patch_size
    current_patches_y = img.size[1] // patch_size
    current_total_patches = current_patches_x * current_patches_y

    # Resize if the current image produces fewer patches than a 256x256 image
    if current_total_patches < target_patches:
        img = img.resize((max(256, img.size[0]), max(256, img.size[1])))

    

    patch_size = 32
    grayscale_patches = []
    color_patches = []
    for i in range(0, img.height, patch_size):
        for j in range(0, img.width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = np.asarray(img.crop(box))
            grayscale_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            grayscale_patches.append(grayscale_patch.astype(np.int32))
            color_patches.append(patch)

    return grayscale_patches, color_patches

def get_l1(v):
    return np.sum(np.abs(v[:, :-1] - v[:, 1:]))

def get_l2(v):
    return np.sum(np.abs(v[:-1, :] - v[1:, :]))

def get_l3l4(v):
    l3 = np.sum(np.abs(v[:-1, :-1] - v[1:, 1:]))
    l4 = np.sum(np.abs(v[1:, :-1] - v[:-1, 1:]))
    return l3 + l4

def get_pixel_var_degree_for_patch(patch):
    l1 = get_l1(patch)
    l2 = get_l2(patch)
    l3l4 = get_l3l4(patch)
    return l1 + l2 + l3l4
def duplicate_to_minimum_sorted(patches, variances, min_count=64):
    """
    Ensures at least min_count patches by duplicating existing ones, while maintaining sorting by variance.
    """
    if len(patches) < min_count:
        # Pair patches with variances and sort by variance descending
        paired = sorted(zip(patches, variances), key=lambda x: x[1], reverse=True)
        while len(paired) < min_count:
            # Shuffle to maintain randomness within the top and bottom elements
            random.shuffle(paired)
            additional_needed = min_count - len(paired)
            paired.extend(paired[:additional_needed])
        # Unzip paired list into patches and variances again
        patches, variances = zip(*paired)
    return list(patches), list(variances)
def extract_rich_and_poor_textures(variance_values, patches):
  
    sorted_indices = np.argsort(variance_values)[::-1]  # Sort indices by variance, descending
    sorted_patches = [patches[i] for i in sorted_indices]
    sorted_variances = [variance_values[i] for i in sorted_indices]
    if(len(patches) < 192):
        threshold = np.mean(variance_values)
        
          # Split into rich and poor based on threshold
        rich_patches = [patch for patch, var in zip(sorted_patches, sorted_variances) if var >= threshold]
        rich_variances = [var for var in sorted_variances if var >= threshold]
        poor_patches = [patch for patch, var in zip(sorted_patches, sorted_variances) if var < threshold]
        poor_variances = [var for var in sorted_variances if var < threshold]
            # Ensure each category has at least 64 patches while maintaining sorted order by variance
        rich_patches, rich_variances = duplicate_to_minimum_sorted(rich_patches, rich_variances, 64)
        poor_patches, poor_variances = duplicate_to_minimum_sorted(poor_patches, poor_variances, 64)
    else:
        num_top_patches = len(patches) // 3
        rich_patches = [patches[i] for i in sorted_indices[:num_top_patches]]
        poor_patches = [patches[i] for i in sorted_indices[-num_top_patches:]]
        
    return rich_patches, poor_patches

def get_complete_image(patches, coloured=True):
    patches = patches[:64]
    
    grid = np.array(patches).reshape((8, 8, 32, 32, 3)) if coloured else np.array(patches).reshape((8, 8, 32, 32))
    rows = [np.concatenate(row_patches, axis=1) for row_patches in grid]
    complete_image = np.concatenate(rows, axis=0)
    return complete_image


def smash_n_reconstruct(input_path, coloured=True):
    grayscale_patches, color_patches = img_to_patches(input_path)
    pixel_var_degree = [get_pixel_var_degree_for_patch(patch) for patch in grayscale_patches]

    if coloured:
        rich_patches, poor_patches = extract_rich_and_poor_textures(pixel_var_degree, color_patches)
    else:
        rich_patches, poor_patches = extract_rich_and_poor_textures(pixel_var_degree, grayscale_patches)

    rich_texture = get_complete_image(rich_patches, coloured)
    poor_texture = get_complete_image(poor_patches, coloured)

    return rich_texture, poor_texture

def apply_high_pass_filter():
    rotated_kernels = []

    for idx, kernel in enumerate(kernels):
        for angle in angles[idx]:
            # Rotate kernel
            rotated_kernel = rotate(kernel, angle, reshape=False)
            # Ensure the kernel is in float32 format
            rotated_kernel = np.round(rotated_kernel).astype(np.float32)
            # Convert to tensor, shape [5, 5]
            tensor_kernel = torch.tensor(rotated_kernel)
            # Unsqueeze and repeat to convert to 3-channel, shape [3, 5, 5]
            tensor_kernel = tensor_kernel.unsqueeze(0).repeat(3, 1, 1)
            rotated_kernels.append(tensor_kernel)

    # Stack all kernels to form a single tensor [num_kernels * num_angles, 3, 5, 5]
    all_kernels = torch.stack(rotated_kernels)

    return all_kernels

# Define the image processing functions


def jpeg_compression(img, quality_range=(70, 100)):
    if random.random() < 0.1:  # 10% probability
        quality = random.randint(*quality_range)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=quality)
        img = Image.open(img_bytes)
    return img


def gaussian_blur(img, sigma_range=(0, 1)):
    if random.random() < 0.1:  # 10% probability
        sigma = random.uniform(*sigma_range)
        img = img.filter(ImageFilter.GaussianBlur(sigma))
    return img


def downsampling(img, scale_range=(0.25, 0.5)):
    if random.random() < 0.1:  # 10% probability
        scale = random.uniform(*scale_range)
        smaller_img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.BICUBIC)
        img = smaller_img.resize((img.width, img.height), Image.BICUBIC)
    return img
def print_model_class_distribution(dataset, indices=None):
    """
    Prints the distribution of model and class combinations in a dataset or a subset.
    
    Args:
        dataset (Dataset): The dataset or subset to analyze.
        indices (list of int, optional): Indices to analyze within the dataset. If None, analyze the entire dataset.
    """
    model_class_counter = Counter()
    total_samples = 0

    # Determine which samples to count
    if indices is None:
        samples_to_count = dataset.samples
    else:
        samples_to_count = [dataset.samples[i] for i in indices]

    # Count each combination of model and class
    for _, class_label, model_name in samples_to_count:
        model_class_counter[(model_name, class_label)] += 1
        total_samples += 1

    # Print the counted distributions
    print(f"Total samples in subset: {total_samples}")
    for (model, class_label), count in model_class_counter.items():
        percentage = (count / total_samples) * 100
        print(f"Model {model}, Class {class_label}: {count} ({percentage:.2f}%)")
def check_data_overlap(subset1, subset2):
    paths1 = {subset1.dataset.file_paths[i] for i in subset1.indices}
    paths2 = {subset2.dataset.file_paths[i] for i in subset2.indices}
    overlap = paths1.intersection(paths2)
    if overlap:
        print(f"Actual data overlap detected with {len(overlap)} items.")
        return overlap
    else:
        print("No actual data overlap detected.")
        return set()