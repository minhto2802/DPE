import os

import torch

import numpy as np
import pandas as pd

DATASETS = [
    "Features",
]


class SubpopDataset:
    SPLITS = {  # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']  # Default, subclasses may override

    def __init__(self, data_dir, split, metadata_path, transform=None, train_attr='yes', subsample_type=None,
                 dynamic_num_samples=False, pre_extracted_feats=None, num_classes=None,
                 multiclass=False, *args, **kwargs):

        df = pd.read_csv(metadata_path)
        self.metadata_path = metadata_path

        for split_idx in df.split.unique():
            df.loc[df.split == split_idx, 'split_indices'] = np.arange(len(df[df.split == split_idx]), dtype=int)

        self.feats = None
        class_target = 'yy' if multiclass else 'y'
        if subsample_type is not None:
            assert split in ['tr', 'va']
            df['subgroup'] = df.y.astype(str) + df.a.astype(str)
            df = subsample(
                df,
                target=class_target if train_attr == 'no' else 'subgroup',
                target_split=0 if split == 'tr' else 1,
                dynamic_num_samples=dynamic_num_samples,
            )

        df = df[df["split"] == (self.SPLITS[split])]
        self.metadata = df

        if pre_extracted_feats is not None:
            try:
                self.feats = pre_extracted_feats[df.split_indices.values.astype('int')]
            except IndexError:
                breakpoint()

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(data_dir, x)).tolist()
        self.y = df[class_target].tolist()
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        self._a = df["a"].tolist()  # original attributes regardless of train_attr
        tmp_df = pd.DataFrame({'classes': np.array(self.y).astype('U'),
                               'attributes': np.array(self._a).astype('U')})
        if ('g' in df.columns) and (df['g'] != -1).all():
            self.g = df['g'].tolist()
        else:
            self.g = (tmp_df.classes + tmp_df.attributes).values.tolist()
        self.transform = transform
        self._count_groups()

        if num_classes is not None:
            self.num_labels = num_classes

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

    @staticmethod
    def label_attr_to_concept(y, a):
        y, a = np.array(y), np.array(a)
        c = np.zeros((len(y), len(np.unique(y)) * len(np.unique(a))))
        c[np.arange(len(y)), y] = 1
        c[np.arange(len(a)), a + len(np.unique(y))] = 1
        return c

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        feat = torch.tensor(0, dtype=torch.long)
        if self.feats is not None:
            feat = torch.tensor(self.feats[i], dtype=torch.float)
            x = torch.tensor(0, dtype=torch.long)
        else:
            if self.transform is not None:
                x = self.transform(self.x[i])
        return i, x, y, a, feat

    def __len__(self):
        return len(self.idx)


def subsample(df: pd.DataFrame,
              target='y',
              target_split: int = 0,
              verbose=False,
              sort_idx=False,
              filter_mask=None,
              num_samples=None,
              dynamic_num_samples=False,
              ) -> pd.DataFrame:
    """

    """
    if target is None:
        return df

    if 'train' in set(df.split):  # special case with ImagenetBG
        splits = {0: 'train', 1: 'val', 2: 'test'}
        target_split = splits[target_split]

    all_num_samples = (df.groupby('split').get_group(target_split)[target].value_counts())
    if num_samples is None:
        num_samples = all_num_samples.min()

    indices = df[df.split != target_split].index.tolist()
    target_indices = []
    for i, subgroup in enumerate(all_num_samples.index):

        idx = (df.groupby('split')
               .get_group(target_split)
               .groupby(target)
               .get_group(subgroup)
               .index.tolist())

        if (
                target == 'subgroup') and dynamic_num_samples:  # equal number of samples in each subgroup, could be different among classes
            df_split = df.groupby('split').get_group(target_split)
            cls = df_split.loc[df_split.subgroup == subgroup].iloc[0].y
            num_samples = df_split.groupby('y').get_group(cls)[target].value_counts().min()

        if filter_mask is not None:
            idx_by_subgroup = (df[df.split == target_split].reset_index()
                               .groupby(target)
                               .get_group(subgroup)
                               .index.tolist())
            sampling_weights = filter_mask[idx_by_subgroup]
            idx = np.array(idx)[np.argsort(sampling_weights)[::-1]]  # [::-1]
        tmp = list(np.random.permutation(idx)[:num_samples])
        target_indices.append(tmp)

    target_indices = np.concatenate(target_indices).reshape((len(target_indices), -1)).T.flatten().tolist()
    indices.extend(target_indices)
    if sort_idx:
        indices = sorted(indices)

    assert len(np.unique(indices)) == len(indices)
    df = df.iloc[indices]

    if verbose:
        print(df[df.split == target_split][target].value_counts())

    return df


class Features(SubpopDataset):
    def __init__(self, *args, **kwargs):
        assert 'pre_extracted_feats' in kwargs and kwargs['pre_extracted_feats'] is not None, \
            "pre_extracted_feats must be provided for Features dataset."
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        i = self.idx[index]
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        feat = torch.tensor(self.feats[i], dtype=torch.float)
        x = torch.tensor(0, dtype=torch.long)  # Dummy value
        return i, x, y, a, feat
