"""
Base classes for biomolecules
"""
import numpy as np
from ..templates.aminoacids import templates_aa, one_to_three_aa
from ..templates.glycans import templates_gl, one_to_three_gl
from ..energy.ff import compute_neighbors, LJ, get_sasa
from ..utils.pdbIO import _dump_pdb
from ..visualization.view3d import _view3d


class Biomolecule():
    """Base class for biomolecules"""

    def __init__(self):
        self.sequence
        self.coords
        self._names
        self._elements
        self._offsets
        self._exclusions

    def __len__(self):
        return len(self.sequence)

    def get_torsionals(self):
        raise NotImplementedError()

    def dump_pdb(self, filename, b_factors=None, to_file=True):
        return _dump_pdb(self, filename, b_factors, to_file)

    def view3d(self):
        return _view3d(self)

    def energy(self, nrg='all', cut_off=6., neighbors=None, params=None, points=1000, selection='all', probe=1.4):
        """
        Compute the internal and solvation energy of a molecule.
        Internal energy is calculated using a pair-wise Lennard-Jones
        potential. Solvation energy is computed using a numerical
        approximation.

        Parameters
        ----------
        
        nrg: str
            Energy to be calculated. It can be 'internal', 'solvation',
            or 'all'. Default 'all'.
        
        For LJ energy
        
        cut_off : float
            Only pair of atoms closer than cut_off will be used to compute the
            energy. Default 6. Only valid when neighbors is None.
        neighbors: set of tuples
            Pairs of atoms used to compute the energy. If None (default) the
            list of neighbors will be computed using a KD-tree (from scipy),
            see ff.compute_neighbors for details.
        
        For Solvation energy
        
        params: dictionary
            {atom: [atom's radius, atom's energy solvatation]}
        points: interger
            Number of point per sphere. Default 1000.
        selection: str
            Atoms selection
        probe : float
            The radius of the solvent. Default 1.4.

        Returns
        ----------
        For LJ energy
        
        energy_lj : float
            molecular energy in Kcal/mol.
        
        For Solvation Energy
        
        energy_solv: float
            The energy of solvation from the atoms selection.
        areas: float
            The total solvent-accessible-surface area from the atoms selection.
        
        All
        nrg_all : float
            total molecular energy in Kcal/mol.
        """
        if nrg=='all' or nrg=='internal':
            coords = self.coords
            if neighbors is None:
                neighbors = compute_neighbors(coords, self._exclusions, cut_off)
            energy_lj = LJ(neighbors, coords, self._elements)
            if nrg=='internal':
                return energy_lj
        
        if nrg=='all' or nrg=='solvation':
            energy_solv, areas = get_sasa(params, points, selection, probe)
            if nrg=='solvation':
                return energy_solv, areas
        
        if nrg=='all':
            nrg_all = energy_lj+energy_solv
            return nrg_all, areas

    def rgyr(self):
        """
        Calculates radius of gyration for a molecule
        ToDo mass-weighted version ¿?

        """
        coords = self.coords
        center = np.mean(coords, 0)
        return np.mean(np.sum((coords - center)**2, 1)) ** 0.5
