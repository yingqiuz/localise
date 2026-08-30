import nibabel as nib
import numpy as np
import os, random
from .batch import get_adj_sparse_kdt, FlattenedCRFBatchTensor, Adjacency
import torch


def _broadcast(arg, n, name):
    """Return `arg` as a list of length n, broadcasting a scalar/None."""
    if isinstance(arg, (list, tuple)):
        if len(arg) != n:
            raise ValueError(
                f'Length of {name} ({len(arg)}) does not match '
                f'the number of subjects ({n}).'
            )
        return list(arg)
    return [arg] * n


def _read_target_list(target_list):
    """Return target_list as a list of filenames, reading a txt file if needed."""
    if isinstance(target_list, (str, os.PathLike)):
        with open(target_list, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return list(target_list)


def load_features(seed, tracts=None, target_list=None, data=None, atlas=None,
                  demean=True, normalise=True, gamma=None, power=None,
                  output_fname=None, adj=None):
    """
    Loads feature matrices and performs several preprocessing steps.

    Parameters
    ----------
    seed : str, Path, or list thereof
        Path to the binary seed (ROI) mask. Features are extracted for the
        non-zero voxels of this mask. A list is treated as multiple subjects,
        in which case `tracts`, `data`, `atlas`, and `output_fname` may each be
        a list of the same length (or a single value shared by all subjects).
    tracts : str or Path, optional
        Path to the folder containing tract-density maps (e.g., seeds_to_*.nii.gz).
        Required if `data` is not provided.
    target_list : list of str, or path to a txt file, optional
        Names of the tract-density files inside `tracts`, or a txt file listing
        one name per line. Required if `data` is not provided.
    data : str or Path, optional
        Path to a pre-saved *.npy feature matrix of shape (n_targets, n_voxels).
        Alternative to `tracts` + `target_list`.
    atlas : str or Path, optional
        Path to a group-average probability map of the structure. If provided,
        it is appended as an additional feature.
    demean : bool, optional
        If True, demean the feature matrix. Defaults to True.
    normalise : bool, optional
        If True, normalise the maximum tract density to 1. Defaults to True.
    gamma : array-like, optional
        The gamma values to use. If not set, defaults to [0].
    power : array-like, optional
        The powers applied to the tract densities. Defaults to [2, 1, 0.5, 0.2].
    output_fname : str or Path, optional
        If set, save the raw feature matrix to this *.npy file.
    adj : tuple, optional
        Pre-computed adjacency (inds1, inds2, n). If not set, computed from the
        seed mask.

    Returns
    -------
    FlattenedCRFBatchTensor
        The tensor of the loaded features (a list thereof if `seed` is a list).

    Raises
    ------
    ValueError
        If neither `data` nor `tracts` is set, or if the loaded data matrix and
        seed mask dimensions do not match.
    """
    if isinstance(seed, (list, tuple)):
        n_subjects = len(seed)
        tracts = _broadcast(tracts, n_subjects, 'tracts')
        data = _broadcast(data, n_subjects, 'data')
        atlas = _broadcast(atlas, n_subjects, 'atlas')
        output_fname = _broadcast(output_fname, n_subjects, 'output_fname')
        return [load_features(seed=s, tracts=t, target_list=target_list,
                              data=d, atlas=a, demean=demean,
                              normalise=normalise, gamma=gamma, power=power,
                              output_fname=o, adj=adj)
                for s, t, d, a, o in zip(seed, tracts, data, atlas, output_fname)]

    if data is None and tracts is None:
        raise ValueError("Please specify either tracts or data.")

    if data is None and target_list is None:
        raise ValueError("Please specify a list of targets if data is not pre-saved.")

    gamma = np.array(gamma).astype(np.float32) if gamma is not None else np.array([0])
    power = np.array(power).astype(np.float32) if power is not None else np.array([2, 1, 0.5, 0.2], dtype=np.float32)

    # load the seed (ROI) mask
    mask = nib.load(str(seed)).get_fdata()
    index = np.where(mask > 0)

    # generate adjacency matrix
    if adj is None:
        inds1, inds2, n = get_adj_sparse_kdt(mask)
    else:
        inds1, inds2, n = adj

    # load data into X
    if data is None:
        target_list = _read_target_list(target_list)
        n_targets = len(target_list)
        X = np.zeros((n_targets, n), dtype=np.float32)
        for k, target in enumerate(target_list):
            X[k, :] = nib.load(os.path.join(str(tracts), target)).get_fdata()[index].astype(np.float32)

        if output_fname is not None:
            np.save(str(output_fname), X)

    else:
        # load pre-saved data
        X = np.load(str(data))
        if X.shape[1] != n:
            raise ValueError("Dimension of the mask and the loaded data matrix do not match. Please check if the loaded data used the same mask.")

    if atlas is not None:
        # load group-average as an additional feature
        ygroup = nib.load(str(atlas)).get_fdata()[index].astype(np.float32)
        ygroup[ygroup < 0.01] = 0
        ygroup /= np.max(ygroup)
        X = np.vstack([np.power(X, el) for el in power] + [ygroup])

    else:
        X = np.vstack([np.power(X, el) for el in power])

    # maximum tract density normalised to 1
    if normalise:
        maxX = np.max(X, axis=1, keepdims=True)
        maxX[ maxX==0 ] = 1
        X /= maxX

    # replace np.nan and np.inf with 0
    X = np.nan_to_num(X)

    if demean:
        X -= np.mean(X, axis=1, keepdims=True)

    return FlattenedCRFBatchTensor(torch.from_numpy(X.T).float(), Adjacency(inds1, inds2, n), K=2, gamma=torch.from_numpy(gamma).float())


def load_labels(seed, labels):
    """
    Loads labels for a given subject.

    This function loads the seed (ROI) mask and label data. It uses the mask to
    index into the label data, and constructs a binary vector where a value is 1
    if the corresponding voxel belongs to the structure, and 0 otherwise.

    Parameters
    ----------
    seed : str, Path, or list thereof
        Path to the binary seed (ROI) mask. A list is treated as multiple
        subjects, in which case `labels` must be a list of the same length.
    labels : str, Path, or list thereof
        Path to the label file (same space as the seed mask).

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 2), where n is the number of non-zero elements in
        the mask. The second column is a binary vector corresponding to the
        label data (1 if the label data > 0, otherwise 0), and the first column
        is its inverse.
    """
    if isinstance(seed, (list, tuple)):
        labels = _broadcast(labels, len(seed), 'labels')
        return [load_labels(seed=s, labels=l) for s, l in zip(seed, labels)]

    mask = nib.load(str(seed)).get_fdata()
    index = np.where(mask > 0)

    y = np.asarray(nib.load(str(labels)).get_fdata()[index] > 0, dtype=np.int32)
    return torch.from_numpy(np.vstack((1 - y, y)).T).float()


def load_data(seed, labels, tracts=None, target_list=None, data=None, atlas=None,
              demean=True, normalise=True, gamma=None, power=None,
              output_fname=None, adj=None):
    """
    This function is a wrapper that loads both features and labels for a given
    subject, and returns them as a tuple.

    The function internally calls the load_features() and load_labels() functions,
    so please refer to their docstrings for more detailed information about each
    parameter.

    Parameters
    ----------
    seed : str, Path, or list thereof
        Path to the binary seed (ROI) mask. A list is treated as multiple
        subjects, in which case `labels`, `tracts`, `data`, `atlas`, and
        `output_fname` may each be a list of the same length (or a single value
        shared by all subjects).
    labels : str, Path, or list thereof
        Path to the label file (same space as the seed mask).
    tracts, target_list, data, atlas, demean, normalise, gamma, power,
    output_fname, adj :
        See load_features().

    Returns
    -------
    tuple
        A tuple containing the loaded features and labels (a list of tuples if
        `seed` is a list). The features are returned as a
        FlattenedCRFBatchTensor, and the labels as a torch Tensor.
    """
    if isinstance(seed, (list, tuple)):
        n_subjects = len(seed)
        labels = _broadcast(labels, n_subjects, 'labels')
        tracts = _broadcast(tracts, n_subjects, 'tracts')
        data = _broadcast(data, n_subjects, 'data')
        atlas = _broadcast(atlas, n_subjects, 'atlas')
        output_fname = _broadcast(output_fname, n_subjects, 'output_fname')
        return [load_data(seed=s, labels=l, tracts=t, target_list=target_list,
                          data=d, atlas=a, demean=demean, normalise=normalise,
                          gamma=gamma, power=power, output_fname=o, adj=adj)
                for s, l, t, d, a, o in zip(seed, labels, tracts, data, atlas, output_fname)]

    features = load_features(seed=seed, tracts=tracts, target_list=target_list,
                             data=data, atlas=atlas, demean=demean,
                             normalise=normalise, gamma=gamma, power=power,
                             output_fname=output_fname, adj=adj)
    labels = load_labels(seed=seed, labels=labels)
    return features, labels


class ShuffledDataLoader():
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        self.index = 0
        random.shuffle(self.data)
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return a different value every time
        return self.data[index]

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def split_data(self, train_proportion):
        # randomize the data
        random.shuffle(self.data)

        # find the split index
        split_idx = round(len(self.data) * train_proportion)
        if split_idx == 0:
            split_idx = 1
        elif split_idx == len(self.data):
            split_idx -= 1

        # split the data
        train_data = self.data[:split_idx]
        validation_data = self.data[split_idx:]

        # create data loaders for train and validation sets
        train_loader = ShuffledDataLoader(train_data)
        validation_loader = ShuffledDataLoader(validation_data)

        return train_loader, validation_loader
