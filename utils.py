import numpy as np

def partial_corr(X, Y, Z):
    # check that X, Y, Z are all 2D arrays
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]
    
    # stack variables into data array
    data = np.hstack([X, Y, Z])
    
    V = np.cov(data, rowvar=False)
    Vi = np.linalg.pinv(V, hermitian=True)  # inverse covariance matrix
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pcor = -1 * (D @ Vi @ D)  # partial correlation matrix
    
    return pcor[0,1]

def pcorr_significance(src, trg, covariates, nulls):
    r_true = partial_corr(src, trg, covariates)
    n_perm = nulls.shape[-1]
    permuted_results = []

    for perm in range(n_perm):
        src_perm = src[nulls[:, perm]]
        permuted_results.append(partial_corr(src_perm, trg, covariates))

    p = (1 + sum(np.abs(permuted_results) > np.abs(r_true))) / (1 + n_perm)
    
    return r_true, p

def significance_marker(p):
    if p <= 0.05:
        return '*'
    else:
        return ' '
    
def load_fs_nifti(path):
    from neuromaps.images import load_nifti
    from neuromaps.parcellate import _array_to_gifti
    """
    path: tuple-of-str
        Tuple containing path to fressurfer NIFTI (.nii) files (left and right hemisphere)
        
    Returns
    -------
    gifti: nibabel.GiftiImage
        GiftiImage object containing data from the NIFTI files
    """
    array = tuple(map(lambda x: load_nifti(x).get_fdata().squeeze(), path))
    array = np.concatenate(array)
    gii = _array_to_gifti(array)
  
    return gii
