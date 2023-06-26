import nibabel as nib
import numpy as np
import os
from joblib import dump, load
from .flatten_batch import get_adj_sparse, FlattenedCRFBatch, FlattenedCRFBatchTensor


DEFAULT_TARGET_LIST = [...]  # fill this with your default target list

def load_features(subject, mask_name, target_path=None, data=None, atlas=None, 
                  target_list=DEFAULT_TARGET_LIST, demean=True, withgroup=False,
                  normalise=True, γ=[0.0], power=[2.0, 1.0, 0.5], 
                  save_features=False, output_fname=None):
    
    if withgroup and atlas is None:
        raise ValueError("If withgroup is set to true, you must specify the file that contains the atlas in the individual space.")
    
    if data is None and target_path is None:
        raise ValueError("Please specify either target_path or data.")
    
    # load mask
    mask = nib.load(os.path.join(subject, mask_name)).get_data()
    index = np.where(mask > 0)
    
    # generate adjacency matrix
    inds1, inds2, n = get_adj_sparse(mask)  # Please implement or import get_adj_sparse() function
    
    n_targets = len(target_list)
    
    # load data into X
    if data is None:
        X = np.zeros((n, n_targets), dtype=np.float32)
        for k in range(n_targets):
            X[:, k] = np.asarray(nib.load(os.path.join(subject, target_path, target_list[k])).get_data()[index], dtype=np.float32)
        
        if save_features:
            dump(X, os.path.join(subject, output_fname))
    
    else:
        # load pre-saved data
        X = load(os.path.join(subject, data))
        if X.shape[0] != n:
            raise ValueError("Dimension of the mask and the loaded data matrix do not match. Please check if the loaded data used the same mask.")
    
    if withgroup:
        # load group-average as an additional feature
        ygroup = nib.load(os.path.join(subject, atlas)).get_data()[index]
        ygroup[ygroup < 0.01] = 0
        ygroup /= np.max(ygroup)
        X = np.hstack([np.power(X, el) for el in power] + [ygroup])
    
    else:
        X = np.hstack([np.power(X, el) for el in power])
    
    # maximum tract density normalised to 1
    if normalise:
        X /= np.max(X, axis=1, keepdims=True)
    
    X[np.isnan(X)] = 0; X[np.isinf(X)] = 0
    if demean:
        X -= np.mean(X, axis=1, keepdims=True)
    
    return X


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

    mask = nib.load(os.path.join(subject, mask_name)).get_data()
    index = np.where(mask > 0)
    
    y = np.asarray(nib.load(os.path.join(subject, label_name)).get_data()[index] > 0, dtype=np.int32)
    return np.vstack((1 - y, y))