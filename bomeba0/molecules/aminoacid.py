"""
Protein class
"""
from ..templates.aminoacids import templates_aa


class aminoacid():
    """Represents a residue. A Residue object stores atoms."""

    def __init__(self, resname, names, coords, occupancies, bfactors):
        self.resname = resname
        self.names = names
        self.coords = coords
        self.occupancies = occupancies
        self.bfactors = bfactors

    def __repr__(self):
        return f'{self.resname}'

    def __getitem__(self, atom):
        if isinstance(atom, str):
            atom_name = atom.upper()
            if atom_name in self.names:
                atom = self.names.index(atom_name)
            elif atom_name in ['SC', 'BB']:
                resinfo = templates_aa[self.resname]
                if atom_name == 'SC':
                    atom = resinfo.sc
                elif atom_name == 'BB':
                    atom = resinfo.bb
            else:
                raise ValueError('Please provide a valid atom name')
        return self.coords[atom]
