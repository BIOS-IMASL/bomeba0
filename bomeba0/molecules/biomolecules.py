"""
Base classes for biomolecules
"""
import numpy as np
from ..templates.aminoacids import templates_aa, one_to_three_aa
from ..templates.glycans import templates_gl, one_to_three_gl
from ..energy.ff import compute_neighbors, LJ
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

    def energy(self, cut_off=6., neighbors=None):
        """
        Compute the internal energy of a molecule using a pair-wise 
        Lennard-Jones potential.

        Parameters
        ----------
        cut_off : float
            Only pair of atoms closer than cut_off will be used to compute the
            energy. Default 6. Only valid when neighbors is None.
        neighbors: set of tuples
            Pairs of atoms used to compute the energy. If None (default) the
            list of neighbors will be computed using a KD-tree (from scipy),
            see ff.compute_neighbors for details.

        Returns
        ----------
        energy : float:
            molecular energy in Kcal/mol

        """
        coords = self.coords
        if neighbors is None:
            neighbors = compute_neighbors(coords, self._exclusions, cut_off)
        energy = LJ(neighbors, coords, self._elements)
        return energy

    def rgyr(self):
        """
        Calculates radius of gyration for a molecule
        ToDo mass-weighted version ¿?

        """
        coords = self.coords
        center = np.mean(coords, 0)
        return np.mean(np.sum((coords - center)**2, 1)) ** 0.5
