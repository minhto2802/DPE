import sys
import math
from sklearn.datasets import make_blobs

sys.path.append('..')
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Creates PyTorch DataLoaders for training and testing.
    """
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets
    training_set = TensorDataset(X_train_tensor, y_train_tensor)
    test_set = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    return training_loader, test_loader, len(training_set), len(test_set)


def train_model(training_loader, test_loader, input_dim, num_classes, device='cuda', epochs=10, lr=5e-2,
                entropic_scale=1, optim_name='adam', show_learning_curves=False):
    """
    Trains the neural network model.
    """
    # Initialize model, loss, and optimizer
    head = IsoMaxPlusLossFirstPart(input_dim, num_classes).to(device)
    criterion = IsoMaxPlusLossSecondPart(entropic_scale=entropic_scale)
    criterion = criterion.to(device)

    if optim_name == 'adam':
        optim = AdamW(head.parameters(), lr=lr)
    else:
        optim = SGD(head.parameters(), lr=lr)

    losses = []
    accuracies = []

    head.train()

    for epoch in range(epochs):
        losses_epoch = []
        acc_list = []
        with tqdm(training_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for x, gt in pbar:
                # x = (x - x.mean(dim=1, keepdims=True)) / x.std(dim=1, keepdims=True)
                x, gt = x.to(device), gt.to(device)

                optim.zero_grad()

                pred = head(x)
                loss = criterion(pred, gt)
                loss.backward()
                optim.step()

                # Calculate accuracy
                acc = (pred.argmax(1) == gt).sum().item() / len(pred)
                acc_list.append(acc)

                losses_epoch.append(loss.item())

                pbar.set_postfix(loss=f'{np.mean(losses_epoch):.3f}',
                                 acc=f'{100 * np.mean(acc_list):.2f}%')

        avg_loss = np.mean(losses_epoch)
        avg_acc = np.mean(acc_list)
        losses.append(avg_loss)
        accuracies.append(avg_acc)

    if show_learning_curves:
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.show()

        # Plot training accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), [acc * 100 for acc in accuracies], marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.grid(True)
        plt.show()

    return head


def normalize_features(X_train, X_test):
    """
    Normalizes features based on training data statistics.
    """
    # return X_train, X_test

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_normed = (X_train - mean) / std
    X_test_normed = (X_test - mean) / std
    return X_train_normed, X_test_normed


def evaluate_model(model, test_loader, device='cuda', set_name='Test'):
    """
    Evaluates the trained model on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    true_pos = []
    with torch.no_grad():
        for x, gt in test_loader:
            # x = (x - x.mean(dim=1, keepdims=True)) / x.std(dim=1, keepdims=True)
            x, gt = x.to(device), gt.to(device)
            pred = model(x)
            predicted = pred.argmax(1)
            true_pos.append(predicted == gt)
            correct += (predicted == gt).sum().item()
            total += gt.size(0)
    accuracy = 100 * correct / total
    # print(f"{set_name} Accuracy: {accuracy:.2f}%")
    model.train()
    return correct, torch.concat(true_pos)


def plot_clusters(X, y, title="Training Data", ax=None):
    """
    Plots clustered data with class labels and attribute ellipses, styled to match the reference figure.

    Parameters:
    - X: Feature matrix (numpy array)
    - y: Labels (numpy array)
    - title: Title of the plot (string)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    # Define colors to match reference
    colors = ['#eab676', '#76b5c5']  # Class 2 (orange), Class 1 (blue)
    scatter = ax.scatter(X[:, 0], X[:, 1],
                         c=[colors[_] for _ in y],
                         alpha=0.6, edgecolor='k',
                         facecolor='w',
                         )

    # Define dashed ellipses for attributes (manually tuned positions/sizes)
    ellipse_specs = [
        {"xy": (0., 0.), "width": 0.14, "height": 0.3, "angle": 0, "label": "Attribute 1", "label_offset": (0.06, 0.15)},
        {"xy": (0, 0), "width": 0.3, "height": 0.1, "angle": 26, "label": "Attribute 2", "label_offset": (0.1, 0.09)},
        {"xy": (0, 0), "width": 0.3, "height": 0.1, "angle": -26, "label": "Attribute 3", "label_offset": (-0.1, 0.09)},
    ]

    # Customize ticks (optional)
    ax.grid(False)

    for spec in ellipse_specs:
        e = patches.Ellipse(spec["xy"], width=spec["width"], height=spec["height"],
                            angle=spec["angle"], edgecolor='gray', facecolor='none',
                            linestyle='dashed', linewidth=1.2)
        ax.add_patch(e)
        ax.text(spec["xy"][0] + spec["label_offset"][0],
                spec["xy"][1] + spec["label_offset"][1],
                spec["label"], fontsize=10, style='italic', ha='center', va='center')

    # Aesthetic cleanup
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    ax.set_aspect('equal', 'box')
    ax.patch.set_alpha(0)

    # Define custom circular legend handles
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Class 1',
               markerfacecolor='#76b5c5', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Class 2',
               markerfacecolor='#eab676', markeredgecolor='k', markersize=8)
    ]

    ax.legend(handles=legend_handles, loc='lower right', frameon=False)

    return ax


def generate_clusters(total_samples=1000, random_state=42):
    np.random.seed(random_state)

    # Define proportions
    major_proportion = 0.96
    minor_proportion = 0.02

    # Number of samples per main cluster
    major_samples = int(total_samples * major_proportion)
    minor_samples = int(total_samples * minor_proportion)

    # Subclusters for major cluster (distinguished by y-axis)
    n_subclusters_major = 2
    major_centers = [
        [0, -0.1],
        [0, 0.1]
    ]
    X_major, y_major = make_blobs(n_samples=major_samples,
                                  centers=major_centers,
                                  cluster_std=0.01,
                                  random_state=random_state)
    # Assign subcluster labels
    y_major_sub = y_major  # 0 or 1 based on y-axis differentiation

    n_subclusters_minor = 2
    minor1_centers = [
        [0.1, 0.05],
        [-0.1, -0.05],
    ]
    X_minor1, y_minor1 = make_blobs(n_samples=minor_samples,
                                    centers=minor1_centers,
                                    cluster_std=0.01,
                                    random_state=random_state + 1)
    y_minor1_sub = y_minor1 + 2  # Labels 2 and 3

    # Subclusters for second minor cluster (distinguished by x-axis)
    minor2_centers = [
        [-0.1, 0.05],
        [0.1, -0.05]
    ]
    X_minor2, y_minor2 = make_blobs(n_samples=minor_samples,
                                    centers=minor2_centers,  # Same as minor1 for consistency
                                    cluster_std=0.01,
                                    random_state=random_state + 2)
    y_minor2_sub = y_minor2 + 4  # Labels 4 and 5

    # Combine all data
    X = np.vstack((X_major, X_minor1, X_minor2))
    subgroups = np.hstack((y_major_sub, y_minor1_sub, y_minor2_sub))
    y = np.where(subgroups % 2 == 0, 0, 1)

    return X, y, subgroups


def generate_balanced_clusters(total_samples=150, random_state=42):
    np.random.seed(random_state)

    # Number of samples per main cluster
    num_samples = int(total_samples / 3)

    # Subclusters for major cluster (distinguished by y-axis)
    n_subclusters_major = 2
    major_centers = [
        [0, -0.1],
        [0, 0.1]
    ]
    X_major, y_major = make_blobs(n_samples=num_samples,
                                  centers=major_centers,
                                  cluster_std=0.01,
                                  random_state=random_state)
    # Assign subcluster labels
    y_major_sub = y_major  # 0 or 1 based on y-axis differentiation

    # Subclusters for first minor cluster (distinguished by x-axis)
    n_subclusters_minor = 2
    minor1_centers = [
        [0.1, 0.05],
        [-0.1, -0.05],
    ]
    X_minor1, y_minor1 = make_blobs(n_samples=num_samples,
                                    centers=minor1_centers,
                                    cluster_std=0.01,
                                    random_state=random_state + 1)
    y_minor1_sub = y_minor1 + 2  # Labels 2 and 3

    # Subclusters for second minor cluster (distinguished by x-axis)
    minor2_centers = [
        [-0.1, 0.05],
        [0.1, -0.05]
    ]
    X_minor2, y_minor2 = make_blobs(n_samples=num_samples,
                                    centers=minor2_centers,  # Same as minor1 for consistency
                                    cluster_std=0.01,
                                    random_state=random_state + 2)
    y_minor2_sub = y_minor2 + 4  # Labels 4 and 5

    # Combine all data
    X = np.vstack((X_major, X_minor1, X_minor2))
    subgroups = np.hstack((y_major_sub, y_minor1_sub, y_minor2_sub))
    y = np.where(subgroups % 2 == 0, 0, 1)

    return X, y, subgroups
