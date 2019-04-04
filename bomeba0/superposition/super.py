"""
Routines related to structural superposition and structural alignment 
"""
import numpy as np

def rmsd_fit(mol0, mol1, selection=None, fit=True, round_to=2):
    """
    Computes the root mean square deviation from two pairs of equivalent atoms after superposition.
    Current implementation is limited to molecules with the same number of atoms.
    
    mol0 : bomeba object
    mol1 : bomeba object
    selection : str
        valid atom name or the strings 'SC' for sidechain or 'BB' for backbone
    fit : bool
        If True (default) mol1 will be rotated and translated to the position that minimize the
        rmsd. If False, the coordinates of the input obeject will not be altered. The value of the 
        rmsd is not affected by this argument. The returned rmsd is always the one obtained after
        superposition.
    """
    xyz0_all = mol0.coords
    xyz1_all = mol1.coords

    if selection is not None:
        xyz0 = np.vstack([aa[selection] for aa in mol0])
        xyz1 = np.vstack([aa[selection] for aa in mol1])
    else:
        xyz0 = mol0.coords
        xyz1 = mol1.coords

    # Translation
    X = xyz0 - xyz0.mean(axis=0)
    Y = xyz1 - xyz1.mean(axis=0)
    # Covariance matrix
    cov_matrix = Y.T @ X
    U, S, Wt = np.linalg.svd(cov_matrix)
    # Optimal Rotation matrix R
    R = U @ Wt
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0.:
        S[-1] = - S[-1]
        Wt[-1] *= - 1
        R = U @ Wt
    if fit:
        #  center the first molecule
        mol0.coords = xyz0_all - xyz0.mean(axis=0)
        #  rotate and translate the second molecule
        mol1.coords = (xyz1_all - xyz1.mean(axis=0)) @ R

    diff = round(np.sum(X ** 2) + np.sum(Y ** 2) - 2.0 * np.sum(S), round_to)
    return (diff / len(X)) ** 0.5
