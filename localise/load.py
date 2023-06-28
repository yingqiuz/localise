import nibabel as nib
import numpy as np
import os
from joblib import dump, load
from .flatten_batch import get_adj_sparse, get_adj_sparse_kdt, FlattenedCRFBatch, FlattenedCRFBatchTensor, Adjacency
import torch


DEFAULT_TARGET_LIST = [...]  # fill this with your default target list

def load_features(subject, mask_name, target_path=None, data=None, atlas=None, 
                  target_list=DEFAULT_TARGET_LIST, demean=True, withgroup=False,
                  normalise=True, gamma=None, power=None, 
                  save_features=False, output_fname=None):
    """
    Loads feature matrices and performs several preprocessing steps.

    Parameters
    ----------
    subject : str
        The name of the subject.
    mask_name : str
        The name of the mask file.
    target_path : str, optional
        The path to the target.
    data : str, optional
        The path to the data.
    atlas : str, optional
        The path to the atlas.
    target_list : list, optional
        A list of targets. Defaults to `DEFAULT_TARGET_LIST`.
    demean : bool, optional
        If True, demean the feature matrix. Defaults to True.
    withgroup : bool, optional
        If True, group-average is loaded as an additional feature. Requires `atlas` to be set. Defaults to False.
    normalise : bool, optional
        If True, normalize the tract density to 1. Defaults to True.
    gamma : array-like, optional
        The gamma values to use. If not set, defaults to an array [0].
    power : array-like, optional
        The power values to use. If not set, defaults to an array [2, 1, 0.5, 0.2].
    save_features : bool, optional
        If True, saves the feature matrix to `output_fname`. Defaults to False.
    output_fname : str, optional
        The output filename to store the feature matrix.

    Returns
    -------
    FlattenedCRFBatchTensor
        The tensor of the loaded features.

    Raises
    ------
    ValueError
        If `withgroup` is True and `atlas` is not set.
        If both `data` and `target_path` are not set.
        If `save_features` is True and `output_fname` is not set.
        If the loaded data matrix and mask dimensions do not match.

    """    
    if withgroup and atlas is None:
        raise ValueError("If withgroup is set to true, you must specify the file that contains the atlas in the individual space.")
    
    if data is None and target_path is None:
        raise ValueError("Please specify either target_path or data.")
    
    if save_features and output_fname is None:
        raise ValueError("Please specify the output filename to store the connectivity feature matrix.")
    
    gamma = np.array(gamma).astype(np.float32) if gamma is not None else np.array([0])
    power = np.array(power).astype(np.float32) if power is not None else np.array([2, 1, 0.5, 0.2], dtype=np.float32)

    # load mask
    mask = nib.load(os.path.join(subject, mask_name)).get_fdata()
    index = np.where(mask > 0)
    
    # generate adjacency matrix
    inds1, inds2, n = get_adj_sparse_kdt(mask)
    
    n_targets = len(target_list)
    
    # load data into X
    if data is None:
        X = np.zeros((n_targets, n), dtype=np.float32)
        for k in range(n_targets):
            X[k, :] = nib.load(os.path.join(subject, target_path, target_list[k])).get_fdata()[index].astype(np.float32)
        
        if save_features:
            #dump(X, os.path.join(subject, output_fname))
            np.save(os.path.join(subject, output_fname), X)
    
    else:
        # load pre-saved data
        #X = load(os.path.join(subject, data))
        X = np.load(os.path.join(subject, data))
        if X.shape[1] != n:
            raise ValueError("Dimension of the mask and the loaded data matrix do not match. Please check if the loaded data used the same mask.")
    
    if withgroup:
        # load group-average as an additional feature
        ygroup = nib.load(os.path.join(subject, atlas)).get_fdata()[index].astype(np.float32)
        ygroup[ygroup < 0.01] = 0
        ygroup /= np.max(ygroup)
        X = np.vstack([np.power(X, el) for el in power] + [ygroup])
    
    else:
        X = np.vstack([np.power(X, el) for el in power])
    
    # maximum tract density normalised to 1
    if normalise:
        X /= np.max(X, axis=1, keepdims=True)
    
    # replace np.nan and np.inf with 0
    X = np.nan_to_num(X)

    if demean:
        X -= np.mean(X, axis=1, keepdims=True)
    
    return FlattenedCRFBatchTensor(torch.from_numpy(X).float(), Adjacency(inds1, inds2, n), K=2, gamma=torch.from_numpy(gamma).float())

def load_labels(subject, mask_name, label_name):
    """
    Loads labels for a given subject.

    This function loads mask and label data for a given subject. It uses the mask to index into the label data,
    and constructs a binary vector where a value is 1 if the corresponding voxel belongs to the structure, and 0 otherwise.
    It returns two stacked arrays, the second is the inversion of the first one.

    Parameters
    ----------
    subject : str
        Path to the subject directory.
    mask_name : str
        Filename of the mask data file.
    label_name : str
        Filename of the label data file.

    Returns
    -------
    numpy.ndarray
        A vertically stacked array of shape (2, n), where n is the number of non-zero elements in the mask. 
        The first row is a binary vector corresponding to the label data (1 if the label data > 0, otherwise 0).
        The second row is the inverse of the first row.
    """

    mask = nib.load(os.path.join(subject, mask_name)).get_fdata()
    index = np.where(mask > 0)
    
    y = np.asarray(nib.load(os.path.join(subject, label_name)).get_fdata()[index] > 0, dtype=np.int32)
    return torch.from_numpy(np.vstack((1 - y, y))).float()