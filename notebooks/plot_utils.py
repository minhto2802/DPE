import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d

import torch.nn.functional as F


def plot_voronoi_extended(vor, ax, xlim, ylim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Center of the Voronoi diagram
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k--', lw=1.5)
        else:
            i = simplex[simplex >= 0][0]  # finite endpoint
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max() * 2

            ax.plot([vor.vertices[i, 0], far_point[0]],
                    [vor.vertices[i, 1], far_point[1]], 'k--', lw=1.5)


def plot_voronoi_with_distances(prototypes, distance_scale, X, y, resolution=500, x_range=(-0.15, 0.15),
                                y_range=(-0.15, 0.15), num_classes=2):
    """
    Plots a Voronoi diagram where the color represents the distance from each point in the grid
    to the nearest prototype.

    Parameters:
    - model: Trained IsoMaxPlusLossFirstPart model
    - X: Numpy array of shape (n_samples, 2), normalized training data
    - resolution: Number of points along each axis for the mesh grid
    """
    # Extract prototypes from the model and convert to NumPy

    # Compute Voronoi diagram
    vor = Voronoi(prototypes)
    # First class prototype
    proto = prototypes

    # Create a mesh grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype('float32')

    # Normalize distances for coloring
    plt.figure(figsize=(8, 8), dpi=150)

    # # Plot Voronoi regions
    plot_voronoi_extended(vor, plt.gca(), xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Force your desired limits AFTER this call
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    custom_colors = ['#eab676', '#76b5c5', '#a1d99b', '#fdcdac', '#bae4b3', '#c6dbef']  # Extend as needed
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.matplotlib.colors.ListedColormap(custom_colors[:num_classes]),
        edgecolor='k',
        alpha=0.7,
        s=50,
        label='Training Data'
    )
    plt.scatter(prototypes[:, 0], prototypes[:, 1],
                c=[0, 1, 0, 1, 0, 1][:len(prototypes)],
                cmap=plt.matplotlib.colors.ListedColormap(['#e28743', '#1e81b0']),
                s=700, edgecolor='k', linewidths=1, marker='*')

    plt.xlabel('Feature 1 (X-axis)')
    plt.ylabel('Feature 2 (Y-axis)')
    plt.title('Voronoi Diagram')
    plt.show()


def plot_distance_to_first_class(model, X, y, resolution=500, x_range=(-0.15, 0.15), y_range=(-0.15, 0.15)):
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
    x_min, x_max = x_range
    y_min, y_max = y_range
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

    plt.figure(figsize=(4, 4), dpi=150)

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
    plt.scatter(prototypes[:, 0], prototypes[:, 1],
                c=['#e28743', '#1e81b0'],
                # cmap='tab10',
                s=500, edgecolor='w', linewidths=1, marker='*')

    plt.xlabel('Feature 1 (X-axis)')
    plt.ylabel('Feature 2 (Y-axis)')
    plt.title('Distance to First Class Prototype')
    plt.grid(False)


def plot_distance_to_first_class_v1(prototypes, distance_scales, X, y, resolution=500, x_range=(-0.15, 0.15),
                                    y_range=(-0.15, 0.15)):
    """
    Plots the data points with background color representing the distance to the first class's prototype.

    Parameters:
    - model: Trained IsoMaxPlusLossFirstPart model
    - X: Numpy array of shape (n_samples, 2), normalized training data
    - y: Numpy array of shape (n_samples,), class labels
    - resolution: Number of points along each axis for the mesh grid
    """
    # Extract prototypes from the model and convert to NumPy
    num_classes = 2

    if num_classes < 1:
        raise ValueError("Model must have at least one prototype.")

    # Define custom colors for classes
    custom_colors = ['#eab676', '#76b5c5', '#a1d99b', '#fdcdac', '#bae4b3', '#c6dbef']  # Extend as needed

    if num_classes > len(custom_colors):
        raise ValueError("Number of classes exceeds number of predefined colors.")

    # First class prototype
    proto = prototypes
    # proto = prototypes[1:2]
    # proto = prototypes[0:1]

    # Create a mesh grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype('float32')
    # grid_points = (grid_points - grid_points.mean(axis=0))/grid_points.std(axis=0)

    # Compute distances to the first class prototype
    distances = torch.abs(distance_scales) * torch.cdist(F.normalize(torch.tensor(grid_points)),
                                                         F.normalize(torch.tensor(proto)),
                                                         p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    distances = distances.argmin(axis=1)
    distances = distances.numpy().reshape(xx.shape)

    # Normalize distances for coloring
    norm = plt.Normalize(distances.min(), distances.max())
    cmap = plt.cm.jet  # Choose a suitable colormap

    fig = plt.figure(figsize=(6, 6), dpi=150)

    # Plot the distance-based coloring
    num_regions = len(np.unique(distances))
    colors = ['#fde9b7', '#bbe8ee', '#f4ce87', '#8edbe5', '#eab676', '#76b5c5']
    plt.imshow(
        distances,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        # cmap=cmap,
        cmap=plt.matplotlib.colors.ListedColormap(colors[:num_regions]),
        norm=norm,
        alpha=0.5,
    )

    custom_colors = ['#eab676', '#76b5c5', '#a1d99b', '#fdcdac', '#bae4b3', '#c6dbef']  # Extend as needed
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.matplotlib.colors.ListedColormap(custom_colors[:num_classes]),
        edgecolor='k',
        alpha=0.7,
        s=50,
        label='Training Data'
    )
    plt.scatter(prototypes[:, 0], prototypes[:, 1],
                c=[0, 1, 0, 1, 0, 1][:len(prototypes)],
                cmap=plt.matplotlib.colors.ListedColormap(['#e28743', '#1e81b0']),
                s=500, edgecolor='w', linewidths=1, marker='*')

    plt.grid(False)

    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()

    # Optionally remove the spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_distance_to_first_class_v2(prototypes, distance_scales, X, y, resolution=500, v_min=0.15, v_max=0.15):
    """
    Plots the data points with background color representing the distance to the first class's prototype.

    Parameters:
    - model: Trained IsoMaxPlusLossFirstPart model
    - X: Numpy array of shape (n_samples, 2), normalized training data
    - y: Numpy array of shape (n_samples,), class labels
    - resolution: Number of points along each axis for the mesh grid
    """
    # Extract prototypes from the model and convert to NumPy
    num_classes = 2

    if num_classes < 1:
        raise ValueError("Model must have at least one prototype.")

    # Define custom colors for classes
    custom_colors = ['#eab676', '#76b5c5', '#a1d99b', '#fdcdac', '#bae4b3', '#c6dbef']  # Extend as needed

    if num_classes > len(custom_colors):
        raise ValueError("Number of classes exceeds number of predefined colors.")

    # First class prototype
    proto = prototypes

    # Create a mesh grid
    x_min, x_max = -v_min, v_max
    y_min, y_max = -v_min, v_max

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype('float32')

    # Compute distances to the first class prototype
    distances = torch.abs(distance_scales) * torch.cdist(F.normalize(torch.tensor(grid_points)),
                                                         F.normalize(torch.tensor(proto)),
                                                         p=2.0, compute_mode="donot_use_mm_for_euclid_dist")

    distances = distances.reshape((distances.shape[0], -1, 2))
    distances = distances.min(axis=1)[0]
    distances = torch.softmax(distances, dim=1)[:, 1]

    distances = distances.numpy().reshape(xx.shape)

    # Normalize distances for coloring
    norm = plt.Normalize(distances.min(), distances.max())
    cmap = plt.cm.jet  # Choose a suitable colormap

    plt.figure(figsize=(6, 6), dpi=150)

    # Plot the distance-based coloring
    plt.imshow(
        distances,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap=cmap,
        norm=norm,
        alpha=1,
    )

    # Plot training data
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.matplotlib.colors.ListedColormap(['#e28743', '#1e81b0']),
        edgecolor='k',
        alpha=0.7,
        s=50,
        label='Training Data'
    )
    plt.scatter(prototypes[:, 0], prototypes[:, 1],
                c=[0, 1, 0, 1, 0, 1][:len(prototypes)],
                cmap=plt.matplotlib.colors.ListedColormap(['#e28743', '#1e81b0']),
                s=700, edgecolor='w', linewidths=1, marker='*')

    plt.xlabel('Feature 1 (X-axis)')
    plt.ylabel('Feature 2 (Y-axis)')
    plt.title('Distance to Blue Prototypes')
    plt.grid(False)
