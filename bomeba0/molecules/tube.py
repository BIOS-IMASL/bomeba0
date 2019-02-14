"""
Base class.
"""


class TestTube:
    """
    This is a "container" class instantiated only once (Singleton)
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(TestTube, cls).__new__(cls)
            cls.molecules = []
            cls.molecules_names = []
        return cls._instance

    def __len__(self):
        """Return the number of molecules."""
        return len(self.molecules)

    def __getitem__(self, idx):
        """Return a molecule by index."""
        return self.molecules[idx]

    def __iter__(self):
        """Iterate over molecules."""
        for molecule in self.molecules:
            yield molecule

    def energy(self):
        """
        Compute the energy of the system.
        ToDo: It should be possible to compute partial energy,
        like without solvent or excluding molecules.
        At the moment this method lives in Protein class
        """
        pass

    def add(self, mol):
        """
        add molecules to TestTube
        """
        molecules = self.molecules
        molecules_names = self.molecules_names
        name = mol.name
        if name in molecules_names:
            print(f'We already have a copy of {name} in the test tube!')
        else:
            molecules.append(mol)
            molecules_names.append(name)

    def remove(self, mol):
        """
        remove molecules from TestTube
        """
        molecules = self.molecules
        if name in molecules:
            molecules.remove(mol)
