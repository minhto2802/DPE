import torch
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d

import torch.nn.functional as F

SET_NAME_DICT = {
    'train_eval': 'Training Set',
    'train': 'Training Set',
    'val': 'Val. Set',
    'test': 'Test Set',
}


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


def plot_voronoi(prototypes, distance_scale, X, y, resolution=500, x_range=(-0.15, 0.15),
                 y_range=(-0.125, 0.125), num_classes=2, set_name='Training', note=''):
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
    plt.figure(figsize=(7, 9), dpi=150)

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
                s=500, edgecolor='k', linewidths=1, marker='*')

    plt.title(f'Voronoi Diagram\n({set_name} set{note})')
    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()
    ax.set_aspect('equal', 'box')


def plot_distance_to_first_class(model, X, y, resolution=500, x_range=(-0.15, 0.15), y_range=(-0.15, 0.15),
                                 ax=None, set_name='Training', note=''):
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

    if ax is None:
        plt.figure(figsize=(4, 4), dpi=150)
        ax = plt.gca()

    # Plot the distance-based coloring
    ax.imshow(
        distances,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap=cmap,
        norm=norm,
        alpha=0.9,
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
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.matplotlib.colors.ListedColormap(custom_colors[:num_classes]),
        edgecolor='k',
        alpha=0.7,
        s=50,
        label='Training Data'
    )
    ax.scatter(prototypes[:, 0], prototypes[:, 1],
               c=['#e28743', '#1e81b0'],
               # cmap='tab10',
               s=500, edgecolor='w', linewidths=1, marker='*')

    ax.grid(False)

    # Aesthetic cleanup
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Distance to Class 1 Prototype\n({set_name} set{note})')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    ax.set_aspect('equal', 'box')
    ax.patch.set_alpha(0)


def plot_distance_to_first_class_v1(prototypes, distance_scales, X, y, resolution=500, x_range=(-0.15, 0.15),
                                    y_range=(-0.15, 0.15), ax=None, set_name='Test', note=''):
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

    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = plt.gca()

    # Plot the distance-based coloring
    num_regions = len(np.unique(distances))
    colors = ['#fde9b7', '#bbe8ee', '#f4ce87', '#8edbe5', '#eab676', '#76b5c5']
    ax.imshow(
        distances,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        # cmap=cmap,
        cmap=plt.matplotlib.colors.ListedColormap(colors[:num_regions]),
        norm=norm,
        alpha=0.5,
    )

    custom_colors = ['#eab676', '#76b5c5', '#a1d99b', '#fdcdac', '#bae4b3', '#c6dbef']  # Extend as needed
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.matplotlib.colors.ListedColormap(custom_colors[:num_classes]),
        edgecolor='k',
        alpha=0.7,
        s=50,
        label='Training Data'
    )
    ax.scatter(prototypes[:, 0], prototypes[:, 1],
               c=[0, 1, 0, 1, 0, 1][:len(prototypes)],
               cmap=plt.matplotlib.colors.ListedColormap(['#e28743', '#1e81b0']),
               s=500, edgecolor='w', linewidths=1, marker='*')

    plt.grid(False)

    # Aesthetic cleanup
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Distance to Class 1 Prototype\n({set_name} set{note})')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    ax.set_aspect('equal', 'box')
    ax.patch.set_alpha(0)


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


# Function to plot bars with numbers and percentages on top
def plot_bar_with_percentage(ax, data, title):
    # Labels for bars
    labels = ['A', 'B', 'C', 'D']

    # Define colors and textures
    textures = ['/', '/']  # Bars 1 & 2 have the same texture
    textures2 = ['\\', '\\']  # Bars 3 & 4 have the same texture
    colors = ['#eab676', '#2596be']  # Land and Water colors

    total = sum(data)
    for i, (value, texture, color) in enumerate(
            zip(data, textures + textures2, [colors[0], colors[1], colors[0], colors[1]])
    ):
        bar = ax.bar(labels[i], value, hatch=texture, color=color, edgecolor='black', width=0.5)
        # Add value and percentage label
        percentage_text = f"{value} ({value / total * 100:.1f}%)"
        ax.text(
            i, value + max(data) * 0.02, percentage_text, ha='center', va='bottom'
        )
    ax.set_title(title, fontsize=20, pad=15, weight='bold')  # Larger font size for titles
    ax.set_ylabel('# Samples', fontsize=16)  # Increased font size for y-label

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # Remove y-ticks
    ax.grid(False)

    # Remove the box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)  # Thicker y-axis
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)  # Thicker x-axis
    ax.spines['bottom'].set_color('black')


def plot_distributions(datasets, group_dict=None, fig_size=(12, 6), dpi=150,
                       title='Subpopulation Distribution', set_name_dict=None):
    # Increase font size globally
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})

    # Create the figure and axes
    fig, axes = plt.subplots(len(datasets.keys()), 1, figsize=fig_size, sharex=True, dpi=dpi)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Get number of samples per class (unavailable subgroup annotation) or per subgroup
    counts = []
    for set_name in datasets.keys():
        counts.append(np.unique(datasets[set_name].g, return_counts=True)[1])

    # Plot each chart with percentages
    set_name_dict = SET_NAME_DICT if set_name_dict is None else set_name_dict
    for i, set_name in enumerate(datasets.keys()):
        plot_bar_with_percentage(axes[i], counts[i], set_name_dict[set_name])

    # Set x-axis label
    if group_dict is not None:
        axes[-1].tick_params(axis='x', which='both', labelbottom=True)
        axes[-1].set_xticks(range(len(group_dict)))
        axes[-1].set_xticklabels(group_dict.values(), fontsize=16)

    # Adjust layout for clarity
    plt.suptitle(title, fontsize=22, weight='bold')
    plt.tight_layout()


def show_examples(datasets, group_dict=None, set_name='val'):
    groups = np.unique(datasets[set_name].g)
    _, axes = plt.subplots(1, len(groups), figsize=(12, 4))
    for i, g in enumerate(groups):
        idx = np.random.choice(np.where(np.array(datasets[set_name].g) == g)[0], 1)[0]
        img = datasets[set_name][idx][1]
        img = UnNormalize()(img).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        if group_dict is not None:
            axes[i].set_title(group_dict[str(g)], fontsize=18)
    plt.suptitle(f'Examples of {SET_NAME_DICT[set_name]}', fontsize=22, weight='bold')
    plt.tight_layout()


class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot_metrics(_df, metric='Worst Group Accuracy', ax=None, show_legend=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _ = sns.lineplot(
        _df,
        x='Number of Prototypes',
        y=metric,
        hue='Diversification Strategy',
        style='Diversification Strategy',
        palette=['#dd5355', '#fe994a', '#438ac3'],
        marker='o',
        ax=ax,
    )
    ax.set_title(metric, fontsize=14)
    ax.get_legend().remove()

    ax.set_ylabel('')
    ax.set_ylim([int(_df[metric].min()), np.ceil(_df[metric].max())])

    if show_legend:
        fig = ax.get_figure()
        plt.suptitle(f'Performance on Waterbirds Dataset', weight='bold', fontsize=18)

        # Extract handles and labels from one axis
        handles, labels = ax.get_legend_handles_labels()
        # Add shared legend to the figure
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            frameon=True,
            fontsize=14,
        )
        plt.tight_layout()

    ax.set_xticks(range(3, _df['Number of Prototypes'].max() + 1, 3))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)


def dict_to_df(_metrics):
    _df = []
    for k in _metrics.keys():
        _df.append(pd.DataFrame({
            'Worst Group Accuracy': np.array(_metrics[k][0]) * 100,
            'Balanced Accuracy': np.array(_metrics[k][1]) * 100,
            'Diversification Strategy': k,
            'Number of Prototypes': np.arange(1, len(_metrics[k][0]) + 1),
        }))
    _df = pd.concat(_df)
    return _df


def show_erm_per_group_accuracy(results, groups_dict, dataset_name='Waterbirds'):
    """
    Plot the per-group accuracy for the ERM model.
    :param results: the last output of the function eval_metrics()
    :param groups_dict:
    :param dataset_name:
    :return:
    """
    df_erm = pd.DataFrame(
        {'Accuracy': [np.round(results['per_group'][k]['accuracy'] * 100) for k in groups_dict.keys()]})
    df_erm['Group'] = [groups_dict[k] for k in groups_dict.keys()]
    _, ax = plt.subplots(figsize=(8, 2), dpi=200)
    sns.barplot(df_erm, x='Group', y='Accuracy', palette=sns.color_palette('Blues', len(df_erm['Group'].unique())),
                hue='Group', width=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(60, 101)
    ax.set_xlabel('')
    ax.set_yticks(range(60, 101, 10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.title(f'ERM Per Group Accuracy on {dataset_name} Test Set', fontsize=12, weight='bold')