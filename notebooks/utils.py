import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d

sys.path.append('..')
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('..')
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
                entropic_scale=1, optim_name='adam'):
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
    print(f"{set_name} Accuracy: {accuracy:.2f}%")
    model.train()
    return correct, torch.concat(true_pos)


def plot_distance_to_first_class(model, X, y, resolution=500):
    """
    Plots the data points with background color representing the distance to the first class's prototype.

    Parameters:
    - model: Trained IsoMaxPlusLossFirstPart model
    - X: Numpy array of shape (n_samples, 2), normalized training data
    - y: Numpy array of shape (n_samples,), class labels
    - resolution: Number of points along each axis for the mesh grid
    """
    # Extract prototypes from the model and convert to NumPy
    distance_scale = model.distance_scale.detach().cpu()
    prototypes = model.prototypes.detach().cpu().numpy()
    num_classes = prototypes.shape[0]

    if num_classes < 1:
        raise ValueError("Model must have at least one prototype.")

    # Define custom colors for classes
    custom_colors = ['#eab676', '#76b5c5', '#a1d99b', '#fdcdac', '#bae4b3', '#c6dbef']  # Extend as needed

    if num_classes > len(custom_colors):
        raise ValueError("Number of classes exceeds number of predefined colors.")

    # First class prototype
    proto = prototypes

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype('float32')

    # Compute distances to the first class prototype
    distances = torch.abs(distance_scale) * torch.cdist(F.normalize(torch.tensor(grid_points)),
                                                        F.normalize(torch.tensor(proto)),
                                                        p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    distances = torch.softmax(distances, dim=1)[:, 1]
    distances = distances.numpy().reshape(xx.shape)

    # Normalize distances for coloring
    norm = plt.Normalize(distances.min(), distances.max())
    cmap = plt.cm.jet  # Choose a suitable colormap

    plt.figure(figsize=(8, 8), dpi=150)

    # Plot the distance-based coloring
    plt.imshow(
        distances,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap=cmap,
        norm=norm,
        alpha=0.8,
    )

    # Optionally, plot Voronoi diagram or decision boundaries
    if num_classes >= 3:
        # Compute Voronoi diagram
        try:
            vor = Voronoi(prototypes)
            voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False,
                            line_colors='black', line_width=2, line_alpha=0.6, point_size=2)
        except Exception as e:
            print(f"Error creating Voronoi diagram: {e}")

    # Plot training data
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.matplotlib.colors.ListedColormap(custom_colors[:num_classes]),
        edgecolor='k',
        alpha=0.7,
        s=50,
        label='Training Data'
    )
    plt.scatter(prototypes[:, 0], prototypes[:, 1], c=['#e28743', '#1e81b0'],
                cmap='tab10', s=500, edgecolor='w', linewidths=1, marker='*')

    plt.xlabel('Feature 1 (X-axis)')
    plt.ylabel('Feature 2 (Y-axis)')
    plt.title('Distance to First Class Prototype')
    plt.grid(False)


def plot_clusters(X, y, title="Clustered Data", ax=None):
    """
    Plots a scatter plot of the clusters and prints axis information.

    Parameters:
    - X: Feature matrix (numpy array)
    - y: Labels (numpy array)
    - title: Title of the plot (string)
    """
    if ax is None:
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = plt.gca()

    colors = ['#eab676', '#76b5c5']
    scatter = ax.scatter(X[:, 0], X[:, 1],
                         c=[colors[_] for _ in y],
                         alpha=0.6, edgecolor='k',
                         facecolor='w',
                         )

    # Customize ticks (optional)
    ax.set_xticks(np.linspace(X[:, 0].min() - X[:, 0].min() * 0.1, X[:, 0].max() + X[:, 0].min() * 0.1, 5))
    ax.set_yticks(np.linspace(X[:, 1].min() - X[:, 1].min() * 0.1, X[:, 1].max() + X[:, 1].min() * 0.1, 5))

    # Add grid
    ax.grid(False)

    # Retrieve and print axis limits
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the background of the axes to transparent (optional)
    ax.patch.set_alpha(0)

    # Optionally remove the spines for a cleaner look
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    ax.set_title(title)

    # plt.show()
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


# class IsoMaxPlusLossFirstPart(nn.Module):
#     """This part replaces the model classifier output layer nn.Linear()"""
#
#     def __init__(self, num_features, num_classes, temperature=1.0, const_init=0):
#         super(IsoMaxPlusLossFirstPart, self).__init__()
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.temperature = temperature
#         self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
#         self.distance_scale = nn.Parameter(torch.Tensor(1))
#         nn.init.normal_(self.prototypes, mean=const_init, std=1e-2)
#         nn.init.constant_(self.distance_scale, 1.0)
#
#     def forward(self, features):
#         distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features), F.normalize(self.prototypes),
#                                                                  p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
#         logits = -distances
#         # The temperature may be calibrated after training to improve uncertainty estimation.
#         return logits / self.temperature
#
#
# class IsoMaxPlusLossSecondPart(nn.Module):
#     """This part replaces the nn.CrossEntropyLoss()"""
#
#     def __init__(self, entropic_scale=20.0, reduction='none'):
#         super(IsoMaxPlusLossSecondPart, self).__init__()
#         self.entropic_scale = entropic_scale
#         self.reduction = reduction
#
#     def forward(self, logits, targets, reduction=None, debug=False):
#         #############################################################################
#         #############################################################################
#         """Probabilities and logarithms are calculated separately and sequentially"""
#         """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
#         #############################################################################
#         #############################################################################
#         self.reduction = reduction if reduction is not None else self.reduction
#         targets = targets.argmax(1) if targets.ndim == 2 else targets  # added
#         distances = -logits
#         probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
#         probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
#         loss = -torch.log(probabilities_at_targets)
#
#         if not debug:
#             if reduction == 'none':
#                 return loss
#             return loss.mean()
#         else:
#             targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
#             intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
#             inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
#             intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
#             inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
#             return loss, 1.0, intra_distances, inter_distances
