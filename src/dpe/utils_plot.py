import torch
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F

SET_NAME_DICT = {
    'train_eval': 'Training Set',
    'train': 'Training Set',
    'val': 'Val. Set',
    'test': 'Test Set',
}

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


def show_examples(datasets, group_dict=None, set_name='val', is_bk=False, *args, **kwargs):
    groups = np.unique(datasets[set_name].g)
    _, axes = plt.subplots(1, len(groups), figsize=(12, 4))
    for i, g in enumerate(groups):
        idx = np.random.choice(np.where(np.array(datasets[set_name].g) == g)[0], 1)[0]
        img = datasets[set_name][idx][1]
        if not is_bk:
            img = UnNormalize()(img).permute(1, 2, 0).cpu().numpy()
            cmap = None
        else:
            cmap = 'gray'
        axes[i].imshow(img, cmap=cmap)
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


def plot_metrics(_df, metric='Worst Group Accuracy', ax=None, show_legend=False, dataset_name='Waterbirds'):
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
        plt.suptitle(f'Ensemble Performance on {dataset_name}', weight='bold', fontsize=18)

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


def show_erm_per_group_accuracy(results, groups_dict, dataset_name='Waterbirds', palette=None, ylim=(60, 101)):
    """
    Plot the per-group accuracy for the ERM model and annotate bars with percentages.
    :param results: the last output of the function eval_metrics()
    :param groups_dict: mapping from internal group indices to readable group names
    :param dataset_name: name of the dataset (used for plot title)
    :param palette: optional color palette
    :return: None
    """
    # Create DataFrame
    df_erm = pd.DataFrame(
        {'Accuracy': [np.round(results['per_group'][k]['accuracy'] * 100) for k in groups_dict.keys()]})
    df_erm['Group'] = [groups_dict[k] for k in groups_dict.keys()]

    # Plotting
    _, ax = plt.subplots(figsize=(8, 2), dpi=200)
    palette = sns.color_palette('Blues', len(df_erm['Group'].unique())) if palette is None else palette
    barplot = sns.barplot(df_erm, x='Group', y='Accuracy', palette=palette, hue='Group', width=0.5)

    # Add percentage labels
    for p in barplot.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0f}%',
                    (p.get_x() + p.get_width() / 2., height + 0.5),
                    ha='center', va='bottom', fontsize=9)

    # Aesthetics
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_xlabel('')
    ax.set_yticks(range(*ylim, 10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.title(f'ERM Per Group Accuracy on {dataset_name} Test Set', fontsize=12, weight='bold')
    plt.legend([], [], frameon=False)  # Hide legend since hue duplicates x-axis
    plt.tight_layout()
    plt.show()


def plot_per_group_accuracy(metrics, groups_dict, title=None, ylim=(70, 101), figsize=(16, 4), dpi=200):
    """
    Plots per-group accuracy for each diversification strategy and annotates bars.

    Parameters:
        metrics (dict): A dictionary where each key corresponds to a diversification strategy,
                        and each value is a list of metric dictionaries. The last item in the list
                        is used for plotting.
        groups_dict (dict): Mapping from group identifier to group name.
        title (str): Title of the plot.
        ylim (tuple): Limits for the y-axis.
        figsize (tuple): Size of the figure.
        dpi (int): Resolution of the figure.
    """
    results = {k: metrics[k][-1] for k in metrics.keys()}

    df2 = []
    for k, val in results.items():
        per_group_acc = val['per_group']
        acc_dict = {g: per_group_acc[g]['accuracy'] * 100 for g in per_group_acc}
        df_tmp = pd.DataFrame({
            'Accuracy': acc_dict.values(),
            'Group': [groups_dict[kk] for kk in acc_dict.keys()],
            'Diversification Strategy': k
        })
        df2.append(df_tmp)

    df2 = pd.concat(df2, ignore_index=True)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.barplot(
        data=df2,
        x='Diversification Strategy',
        y='Accuracy',
        hue='Group',
        palette=sns.color_palette('Blues', len(groups_dict))
    )
    ax.set_ylim(*ylim)
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=8)

    # Move legend below the x-axis label
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(groups_dict),
        frameon=True
    )
    plt.title(title or 'Per-group Accuracy across Diversification Strategies', weight='bold')
    plt.tight_layout()
    plt.show()
