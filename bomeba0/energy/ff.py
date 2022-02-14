"""
Draft of a force-field.
At this point only a Lennard Jones and a solvation term are implemented.

Functions assign_atom_types(), assign_params(), get_neighbors() and, 
get_sasa() adapted from
https://github.com/BIOS-IMASL/Azahar/blob/master/Azahar/energy.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from numba import jit
from ..utils.geometry import dist
from ..utils.constants import par_s_ij, par_eps_ij

try:
    import openbabel as ob
except ImportError:
    pass

def assign_atom_types(selection='all'):
    """
    Assign properties at each atom in the selection.
    
    Using openbabel-2.4.1
    read http://openbabel.org/dev-api/classOpenBabel_1_1OBAtom.shtml#ae09ed28481ac044dab3f31c8605b44a9
    for available functions provided by openbabel to extract atom properties.
    There is a GetType() function but is more limited.
    
    Parameters
    -----------
    selection: str
        Atoms selection
    
    Returns
    --------
    atom_types: list of tuples, a tuple for each description of an atom
        [(at_index, elem, heavy_nb, in_ring, is_arom,
          ring_membership, neighbors)]
    """
    atom_types = []
    pdb_string = selection.dump_pdb(filename=False, b_factors=None, to_file=False)
    mol = ob.OBMol()
    obconversion = ob.OBConversion()
    obconversion.SetInAndOutFormats('pdb', 'pdb')
    obconversion.ReadString(mol, pdb_string)
    rings = mol.GetSSSR()
    for at in ob.OBMolAtomIter(mol):
        ring_member = [ring.IsMember(at) for ring in rings]
        neighbors = [neighbor.GetAtomicNum()
                     for neighbor in ob.OBAtomAtomIter(at)]
        atom_types.append(
            (at.GetIndex(),
             at.GetAtomicNum(),
             at.GetHvyValence(),
             any(ring_member),
                at.IsAromatic(),
                at.MemberOfRingCount(),
                (neighbors)))
    return atom_types


def assign_params(atom_types):
    """
    Assign solvation parameters for each atom.
    For now, simplified from: DOI: 10.1002/prot.340140112
    
    SRA_model = {'atom_types': (Radius, theta)}
    Radius: Hydrated radius in Angstroms
    theta: Atomic solvatation parameter (kcal/mol/A²)
    
    Parameters
    ----------
    atom_types: list of tuples, a tuple for each description of atoms
        [(at_index, elem, heavy_nb, in_ring, is_arom,
          ring_membership, neighbors)]
    
    Result
    --------
    params: Dictionary of atoms index with the corresponding Radius and theta
        {'atom_index': (Radius, theta)}
    """
    SRA_model = {'hydroxyl_carboxyl H': (2.85, -0.0487),
                 'amine_amide H': (2.85, -0.0487),
                 'thiol H': (2.85, -0.0487),
                 
                 'aliphatic CH3': (0.946, 0.0676),
                 'aliphatic CH2': (0.946, 0.0676),
                 'aliphatic CH': (0.946, 0.0676),
                 'aliphatic_alicyclic C': (0.946, 0.0676),
                 'alicyclic CH2': (0.946, 0.0676),
                 'alicyclic CH': (0.946, 0.0676),
                 'aromatic CH': (0.946, 0.0676),
                 'aromatic C': (0.946, 0.0676),
                 'aromatic C of fused ring': (0.946, 0.0676),
                 'aromatic CH of fused ring': (0.946, 0.0676),
                 'carbonyl_carboxylic C': (0.946, 0.0676),
                 
                 'N primary amine': (4.10, -0.0225),
                 'N secondary amine': (4.10, -0.0225),
                 'N cyclic amine': (4.10, -0.0225),
                 'aromatic N': (4.10, -0.0225),
                 'N amide': (4.10, -0.0225),
                 
                 'ether_hydroxyl O': (2.83, -0.0282),
                 'carboxylic O': (2.83, -0.0282),
                 'carbonyl O': (2.83, -0.0282),
                 'amide carbonyl O': (2.83, -0.0282),
                 
                 'thiol or sulfide S': (7.37, -0.0020)}
    params = {}
    for atom in atom_types:
        at_index, elem, heavy_nb, in_ring, is_arom, ring_membership, neighbors = atom
        if elem == 1:
            if (neighbors[0] == 6 or neighbors[0] == 8):
                params[at_index] = SRA_model['hydroxyl_carboxyl H']
            elif neighbors[0] == 7:
                params[at_index] = SRA_model['amine_amide H']
            elif neighbors[0] == 16:
                params[at_index] = SRA_model['thiol H']
        elif elem == 6:
            if neighbors.count(1) == 3:
                params[at_index] = SRA_model['aliphatic CH3']
            elif neighbors.count(1) == 2:
                if in_ring:
                    params[at_index] = SRA_model['alicyclic CH2']
                else:
                    params[at_index] = SRA_model['aliphatic CH2']
            elif neighbors.count(1) == 1:
                if in_ring:
                    params[at_index] = SRA_model['alicyclic CH']
                else:
                    params[at_index] = SRA_model['aliphatic CH']
                if is_arom:
                    if ring_membership > 1:
                        params[at_index] = SRA_model['aromatic CH of fused ring']
                    else:
                        params[at_index] = SRA_model['aromatic CH']
            elif neighbors.count(1) == 0:
                if is_arom:
                    if ring_membership > 1:
                        params[at_index] = SRA_model['aromatic C of fused ring']
                    else:
                        params[at_index] = SRA_model['aromatic C']
                else:
                    params[at_index] = SRA_model['aliphatic_alicyclic C']
            elif neighbors.count(8) > 1:
                params[at_index] = SRA_model['carbonyl_carboxylic C']
        elif elem == 7:
            if neighbors.count(1) == 2:
                if is_arom:
                    params[at_index] = SRA_model['N cyclic amine']
                else:
                    params[at_index] = SRA_model['N primary amine']
            elif neighbors.count(1) == 1:
                params[at_index] = SRA_model['N secondary amine']
            elif is_arom:
                params[at_index] = SRA_model['aromatic N']
            # elif # N de laamide
                #params.append(SRA_model['aromatic N'])
        elif elem == 8:
            if len(neighbors) == 2:
                params[at_index] = SRA_model['ether_hydroxyl O']
            else:
                params[at_index] = SRA_model['carboxylic O']
            # C=O
            # C=O of ester
            # O of amide
        if elem == 16:
            params[at_index] = SRA_model['thiol or sulfide S']
    return params


def get_neighbors(coord, probe, k, params):
    """
    Returns list of index of neighbors.
    
    TODO: Combine with compute_neighbors()
    
    Parameters
    ----------
    coord : Array (n,3)
        Cartesian coordinates
    probe : float
        Radius of the solvent
    k: int
        Atoms index from which to compute neighbors
    """
    dist = cdist(coord, coord, metric='euclidean')
    neighbor_indices = []
    radius = params[k][0] + probe * 2
    for key, values in params.items():
        if dist[key, k] < radius + values[0]:
            neighbor_indices.append(key)
    return neighbor_indices


def get_sasa(params, points, selection='all', probe=1.4):
    """
    Returns the solvent-accessible-surface area and empirical solvation term.
    
    Parameters
    ----------
    params: dictionary
        {atom: [atom's radius, atom's energy solvatation]}
    points: interger
        Number of points per sphere
    selection: str
        Atoms selection
    probe : float
        The radius of the solvent
    
    Return
    ------
    energy: float
        The energy of solvation from the atoms selection
    areas: float
        The total solvent-accessible-surface area from the atoms selection
    """
    # compute the area each point represents
    areas = []
    energies = []
    coord = selection.coords
    const = 4.0 *(np.pi/points)
    for key, values in params.items():
        # scale the points to the correct radius
        radius = values[0] + probe
        points_scaled = (coord[key] + points * radius).reshape(1,-1)
        # get all the indices of neighbors of the i residue
        neighbors = get_neighbors(coord, probe, key, params)
        # compute the distance between points and neighbors
        d_matrix = cdist(points_scaled, coord[neighbors],
                         metric='euclidean')
        # create a matrix and store the vdW radii for each neighbor
        nb_matrix = np.zeros((len(points_scaled), len(neighbors)))
        for nb_i, nb in enumerate(neighbors):
            nb_matrix[:, nb_i] = values[0]
        # compute the number of buried points, we have to be carefull
        # because we have counted how many times a point is buried
        # and we only need how many points are buried
        buried = np.sum(np.sum(d_matrix < nb_matrix + probe, axis=1) > 0)
        exposed = len(points_scaled) - buried
        area_per_atom = const * exposed * radius**2
        energy_per_atom = area_per_atom * values[1]
        areas.append(area_per_atom)
        energies.append(energy_per_atom)
    return sum(energies), sum(areas)


def compute_neighbors(coords, exclusions, cut_off):
    """
    Use a KD-tree (from scipy) to compute the neighbors atoms for each atom.

    Parameters
    ----------
    coords : array (m, 3)
        Cartesian coordinates of a molecule
    exclusions : set of tuples
        Pairs of atoms excluded from the computation of the neighbors
    cut_off : float
        Only pairs of atoms closer than cut_off will be used to compute the
        neighbors.

    Results
    -------
    neighbors: set of tuples
        Pairs of neighbors atoms within a given "cut_off" and excluding
        "exclusions"
    """
    tree_c = cKDTree(coords)
    all_pairs = tree_c.query_pairs(cut_off)
    return all_pairs - exclusions


def LJ(neighbors, xyz, elements):
    """
    Lennard Jones energy term

    .. math::

    LJ_{ij} = \\epsilon \\left [ \\left (\\frac{\\sigma_{ij}}{r_{ij}} \\right)^{12}
     - 2 \\left (\\frac{\\sigma_{ij}}{r_{ij}} \\right)^{6} \\right]

    \\sigma_{ij} is the distance at which the potential reaches its minimum
    \\epsilon_{ij} is the depth of the potential well
    r_{ij} is the distance between the particles

    Parameters
    ----------
    neighbors : set of tuples
        Pairs of neighbors atoms
    xyz : array (m, 3)
        Cartesian coordinates of a molecule
    elements : list of strings
        list of atoms in the molecule, used has key dictionary in
        par_s_ij and par_eps_ij

    Results
    -------
    E_LJ: float
        Lennard Jones energy contribution
    """

    E_vdw = 0.
    for i, j in neighbors:
        key_ij = elements[i] + elements[j]
        # use par_vdw without precomputing values
        #sigma_ij = par_vdw[key_i][0] + par_vdw[key_j][0]
        #epsilon_ij = (par_vdw[key_i][1] * par_vdw[key_j][1])**0.5
        # or use precomputed values
        sigma_ij = par_s_ij[key_ij]
        epsilon_ij = par_eps_ij[key_ij]

        E_vdw += _LJ(xyz, i, j, sigma_ij, epsilon_ij)
    return E_vdw


# convenient function just to speed up computation by 3x
@jit
def _LJ(xyz, i, j, sigma_ij, epsilon_ij):

    r_ij = dist(xyz[i], xyz[j])
    C6 = (sigma_ij / r_ij) ** 6

    return epsilon_ij * (C6 * C6 - 2 * C6)

